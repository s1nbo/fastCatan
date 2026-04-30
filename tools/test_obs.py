#!/usr/bin/env python3
"""Smoke tests for write_obs encoder."""

from __future__ import annotations
import ctypes, re, sys
from pathlib import Path

# Action IDs for setup
SETTLE_BASE, ROAD_BASE = 0, 108
ROLL_DICE = 180
NUM_NODES, NUM_EDGES = 54, 72
NO_PLAYER = 0xFF
PHASE_MAIN = 2
NODE_EMPTY, NODE_SETTLEMENT = 0, 1
NUM_PLAYERS = 4

REPO = Path(__file__).resolve().parents[1]
HDR = (REPO / "include" / "topology.hpp").read_text()
def _parse(text, name):
    m = re.search(rf"\b{re.escape(name)}\s*=\s*\{{\{{(?P<body>.*?)\}}\}}\s*;", text, re.DOTALL)
    rows = re.findall(r"\{\{([^{}]*)\}\}", m.group("body"))
    out = []
    for r in rows:
        v = [int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]+", r)]
        while v and v[-1] == 0xFF: v.pop()
        out.append(tuple(v))
    return tuple(out)

NODE_NODE = _parse(HDR, "node_to_node")
NODE_EDGE = _parse(HDR, "node_to_edge")
EDGE_NODE = _parse(HDR, "edge_to_node")

LIB = REPO / "build" / ("libfastcatan.dylib" if sys.platform == "darwin" else "libfastcatan.so")
if not LIB.exists(): print(f"missing {LIB}; build first"); sys.exit(1)
lib = ctypes.CDLL(str(LIB))
VP, U8, U16, U32, U64, I = (ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint16,
                             ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int)
F = ctypes.c_float
def _b(name, restype, *argtypes):
    f = getattr(lib, name); f.restype = restype; f.argtypes = list(argtypes); return f

create  = _b("fcatan_create",  VP)
destroy = _b("fcatan_destroy", None, VP)
reset_  = _b("fcatan_reset",   None, VP, U64)
step_   = _b("fcatan_step",    U8,   VP, U32)

phase   = _b("fcatan_phase",   U8, VP)
current_player = _b("fcatan_current_player", U8, VP)
node_byte = _b("fcatan_node", U8, VP, I)
edge_byte = _b("fcatan_edge", U8, VP, I)

obs_size  = _b("fcatan_obs_size",  U32)
write_obs = _b("fcatan_write_obs", None, VP, U8, ctypes.POINTER(F))


def node_level(b): return b & 0x03
def node_owner(b): return (b >> 2) & 0x07

def first_legal_settle_initial(env_h):
    for n in range(NUM_NODES):
        if node_level(node_byte(env_h, n)) != NODE_EMPTY: continue
        if any(node_level(node_byte(env_h, nb)) != NODE_EMPTY for nb in NODE_NODE[n]):
            continue
        return n
    return -1

def first_unroaded_settle_road(env_h):
    pl = current_player(env_h)
    for ed in range(NUM_EDGES):
        if edge_byte(env_h, ed) != NO_PLAYER: continue
        for n in EDGE_NODE[ed]:
            b = node_byte(env_h, n)
            if node_level(b) != NODE_SETTLEMENT or node_owner(b) != pl: continue
            already = any(edge_byte(env_h, e2) == pl for e2 in NODE_EDGE[n])
            if not already: return ed
    return -1

def play_initial(env_h):
    while phase(env_h) != PHASE_MAIN:
        step_(env_h, SETTLE_BASE + first_legal_settle_initial(env_h))
        step_(env_h, ROAD_BASE   + first_unroaded_settle_road(env_h))


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


def test_obs_size_constant():
    n = obs_size()
    print(f"  OBS_SIZE = {n}")
    fails = 0
    fails += fail(n > 100, f"OBS_SIZE too small: {n}")
    fails += fail(n < 2000, f"OBS_SIZE too large: {n}")
    return fails


def test_obs_finite_and_nonneg():
    """All obs floats should be finite and non-negative."""
    env = create()
    reset_(env, 42)
    n = obs_size()
    buf = (F * n)()
    write_obs(env, current_player(env), buf)
    fails = 0
    import math
    for i, v in enumerate(buf):
        if not math.isfinite(v):
            fails += fail(False, f"obs[{i}] not finite: {v}")
            break
        if v < 0.0:
            fails += fail(False, f"obs[{i}] negative: {v}")
            break
    destroy(env)
    return fails


def test_obs_changes_after_action():
    """obs should reflect state changes — different obs after a move."""
    env = create()
    reset_(env, 42)
    n = obs_size()
    buf_before = (F * n)()
    buf_after  = (F * n)()
    write_obs(env, current_player(env), buf_before)
    # Place a settlement
    step_(env, SETTLE_BASE + first_legal_settle_initial(env))
    write_obs(env, current_player(env), buf_after)
    diff = sum(1 for i in range(n) if buf_before[i] != buf_after[i])
    destroy(env)
    return fail(diff > 0, f"obs unchanged after settlement build")


def test_obs_pov_differs_per_player():
    """Different POVs should produce different obs (relative-seat encoding)."""
    env = create()
    reset_(env, 42)
    play_initial(env)
    n = obs_size()
    obs_pov = []
    for p in range(NUM_PLAYERS):
        buf = (F * n)()
        write_obs(env, p, buf)
        obs_pov.append(list(buf))
    fails = 0
    pairs_differ = 0
    for i in range(NUM_PLAYERS):
        for j in range(i+1, NUM_PLAYERS):
            if obs_pov[i] != obs_pov[j]: pairs_differ += 1
    fails += fail(pairs_differ >= 5, f"obs identical across POVs ({pairs_differ}/6 differ)")
    destroy(env)
    return fails


def test_obs_determinism():
    """Same state, same POV → same obs."""
    env = create()
    reset_(env, 42)
    play_initial(env)
    n = obs_size()
    a = (F * n)()
    b = (F * n)()
    write_obs(env, 0, a)
    write_obs(env, 0, b)
    diff = sum(1 for i in range(n) if a[i] != b[i])
    destroy(env)
    return fail(diff == 0, f"obs non-deterministic: {diff} bytes differ")


def test_obs_self_pov_marks_self_at_seat0():
    """'is_current' indicator for relseat 0 should equal 1.0 if obs POV is current_player."""
    env = create()
    reset_(env, 42)
    play_initial(env)
    n = obs_size()
    # Per-player block layout: per_player[0] = self block. Slot index of `is_current` is 15.
    # Find self obs
    cp = current_player(env)
    buf = (F * n)()
    write_obs(env, cp, buf)
    # slot 15 in block 0 = is_current for self. Should be 1.0.
    fails = 0
    fails += fail(buf[15] == 1.0, f"self is_current = {buf[15]} (want 1)")
    # Now POV from a non-current player; their seat-0 (themselves) is not current.
    other = (cp + 1) & 3
    write_obs(env, other, buf)
    fails += fail(buf[15] == 0.0, f"non-current self is_current = {buf[15]} (want 0)")
    destroy(env)
    return fails


def main():
    total = 0
    print("== test_obs_size_constant =="); total += test_obs_size_constant()
    print("== test_obs_finite_and_nonneg =="); total += test_obs_finite_and_nonneg()
    print("== test_obs_changes_after_action =="); total += test_obs_changes_after_action()
    print("== test_obs_pov_differs_per_player =="); total += test_obs_pov_differs_per_player()
    print("== test_obs_determinism =="); total += test_obs_determinism()
    print("== test_obs_self_pov_marks_self_at_seat0 =="); total += test_obs_self_pov_marks_self_at_seat0()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} failures")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
