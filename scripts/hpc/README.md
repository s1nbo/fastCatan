# Running fastCatan on the TU Berlin HPC cluster

Step-by-step from zero to a 1000-game evaluation sharded across nodes.
Cluster facts (gateway name, partition names, quotas) drift — where marked
**[verify]**, check the current docs: <https://hpc.tu-berlin.de> (ZECM
HPC documentation) before trusting this file.

## 0. Access (once)

1. HPC access is tied to your TUB account; if `ssh` is refused, request HPC
   access via the ZECM/tubIT portal **[verify: current request form]**.
2. Off-campus: connect the TUB VPN (OpenVPN/WireGuard profile from ZECM)
   first — the gateway is reachable only from the campus network.
3. `ssh <tub-login>@gateway.hpc.tu-berlin.de` **[verify hostname]**.
   Put it in `~/.ssh/config` with your key.

## 1. Recon (5 min, on the login node)

```bash
sinfo -o "%P %l %D %c %m %G"   # partitions, walltime limits, CPUs, mem, GPUs
module avail gcc cmake cuda    # toolchain modules
df -h $HOME /scratch           # quotas; $HOME is usually small
```

Write down: the CPU partition name, the GPU partition name, max walltime,
and whether `/scratch/$USER` (or similar) exists. Fill them into the
`#SBATCH` headers of `eval_array.sbatch` / `train.sbatch`.

## 2. Software (once, on the login node — it has internet; compute nodes may not)

```bash
# Miniforge (conda) into $HOME (or /scratch if $HOME quota is tight):
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh

# Repo:
git clone <your-remote> $HOME/fastCatan && cd $HOME/fastCatan

# Canonical env (py3.12 + pinned deps + native build + verify):
bash scripts/setup.sh                       # CPU
# FASTCATAN_CUDA=cu124 bash scripts/setup.sh  # if you'll use GPU nodes
```

**`-march=native` caveat:** the C++ extension compiles for the CPU it builds
on. If the login node's CPU differs from the compute nodes', rebuild ON a
compute node of your target partition or you risk `SIGILL`:

```bash
srun -p <cpu-partition> --cpus-per-task=4 --time=00:30:00 --pty bash
conda activate catan && cd $HOME/fastCatan
touch include/state.hpp && python -c "import fastcatan"   # editable.rebuild=true recompiles
python scripts/check_env.py
exit
```

If the system GCC is too old for C++23 (need GCC 14): `module load gcc/14`
**[verify version]** before the rebuild, or
`conda install -n catan -c conda-forge cxx-compiler`.

## 3. Smoke (once, interactive)

```bash
srun -p <cpu-partition> --cpus-per-task=4 --time=00:30:00 --pty bash
conda activate catan && cd $HOME/fastCatan
PYTHONHASHSEED=0 PYTHONPATH=EVAL python -m AB.mixed_tournament \
    --games 2 --n-agents 2 --policy mcts --mcts-sims 64 \
    --model-ab-depth 2 --model-ab-prune --ab-depth 2 --ab-prune --seed 1
```

## 4. Big evaluation runs — array sharding

Bridge evals are embarrassingly parallel across games (each game is an
independent seeded catanatron process). `eval_array.sbatch` splits a run
into shards with disjoint seed ranges (shard i covers seeds
`BASE_SEED + i*GAMES_PER_SHARD …`), each writing its own results JSON;
`merge_results.py` pools them into one Wilson CI.

```bash
# 1000-game 2v2 hybrid-MCTS table = 20 shards x 50 games:
sbatch --array=0-19 \
  --export=ALL,GAMES_PER_SHARD=50,BASE_SEED=42,DRIVER=AB.mixed_tournament,EXTRA_ARGS="--n-agents 2 --policy mcts --mcts-sims 512 --model-ab-depth 2 --model-ab-prune --ab-depth 2 --ab-prune" \
  scripts/hpc/eval_array.sbatch

# trades-off ablation arm (append --no-trades to EXTRA_ARGS):
sbatch --array=0-19 \
  --export=ALL,GAMES_PER_SHARD=50,BASE_SEED=42,DRIVER=AB.mixed_tournament,EXTRA_ARGS="--n-agents 2 --policy mcts --mcts-sims 512 --model-ab-depth 2 --model-ab-prune --ab-depth 2 --ab-prune --no-trades" \
  scripts/hpc/eval_array.sbatch

# classic 1v3 gate run, sharded:
sbatch --array=0-19 \
  --export=ALL,GAMES_PER_SHARD=50,BASE_SEED=42,DRIVER=AB.tournament,EXTRA_ARGS="--policy mcts --mcts-sims 512 --model-ab-depth 2 --model-ab-prune --ab-depth 2 --ab-prune" \
  scripts/hpc/eval_array.sbatch

# watch / merge:
squeue -u $USER
python scripts/hpc/merge_results.py "EVAL/AB/results/hpc_<jobid>/*/*.json"
```

Sizing: the 32.5% gate config measured 16.4 s/game with ONE search seat;
a 2v2 table ≈ 2x, 3v1 ≈ 3x search cost. 50-game shards of 2v2 ≈ 30–45 min
each — set `--time` with ≥3x headroom.

NOTE — pooled-vs-single-run equivalence: shard i's games are NOT the same
RNG stream as games `[i*50, (i+1)*50)` of a single 1000-game run (per-game
seeds match, but torch/np global seeding restarts per shard). Statistically
equivalent, not bit-identical. Report pooled numbers as one experiment with
the shard layout stated.

## 5. Long training runs

`train.sbatch` is a single-node template (`CMD=` passthrough). Self-play
tops out ~1700 fps regardless of `num_envs` (CPU opponent inference
bound) — more cores help the env loop only so far; a GPU helps the bigger
nets. Chain runs past the walltime limit with checkpoint-resume:

```bash
jid=$(sbatch --parsable --export=ALL,CMD="python -m models.train_ppo --total-steps 50000000 ..." scripts/hpc/train.sbatch)
sbatch --dependency=afterok:$jid --export=ALL,CMD="python -m models.train_ppo --resume ..." scripts/hpc/train.sbatch
```

Put IL caches / datasets on `/scratch` (purged! rsync results home) and use
`--block-shuffle` for bigger-than-RAM caches.

## 6. Hygiene

- `seff <jobid>` after the first shard: CPU efficiency <50% → request fewer
  cores per task and more array width.
- `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` for post-mortems.
- Everything under `/scratch` is purgeable: `rsync -av` result JSONs back to
  the repo (`EVAL/AB/results/`) and commit them.
- Always export `PYTHONHASHSEED=0` (the sbatch templates do) — catanatron's
  RNG depends on it.
