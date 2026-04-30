// First throughput benchmark for fastCatan.
// Pure C++ random-legal-action loop; measures step_one + compute_mask + reset_one cost.
//
// Build:
//   bash tools/build_bench.sh
// Run:
//   build/bench_step [num_steps]
#include "rules.hpp"
#include "mask.hpp"
#include "rng.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>

using namespace catan;
using clk = std::chrono::high_resolution_clock;

// Pick the n-th set bit in mask, where n = picker.bounded(popcount).
// Returns UINT32_MAX if mask is empty.
static inline uint32_t pick_random_legal(const uint64_t mask[MASK_WORDS],
                                          Xoshiro128& picker) noexcept {
    uint32_t total = 0;
    for (uint32_t i = 0; i < MASK_WORDS; ++i) {
        total += uint32_t(__builtin_popcountll(mask[i]));
    }
    if (total == 0) return UINT32_MAX;

    uint32_t pick = picker.bounded(total);
    for (uint32_t w = 0; w < MASK_WORDS; ++w) {
        uint32_t pc = uint32_t(__builtin_popcountll(mask[w]));
        if (pick < pc) {
            uint64_t v = mask[w];
            for (uint32_t i = 0; i < pick; ++i) v &= v - 1;
            return w * 64 + uint32_t(__builtin_ctzll(v));
        }
        pick -= pc;
    }
    return UINT32_MAX;  // unreachable
}

int main(int argc, char** argv) {
    const uint64_t TOTAL_STEPS = (argc > 1)
        ? std::strtoull(argv[1], nullptr, 10)
        : 1'000'000ULL;

    GameState s;
    BoardLayout b;
    Xoshiro128 picker;
    xoshiro_seed(picker, 0xCAFEBABEULL);

    reset_one(s, b, picker.next());

    uint64_t mask[MASK_WORDS];
    uint64_t step_count = 0;
    uint64_t game_count = 0;
    uint64_t mask_calls = 0;
    uint64_t reset_calls = 1;

    // Warm-up to settle caches / branch predictors.
    for (int i = 0; i < 1000; ++i) {
        compute_mask(s, b, mask);
        ++mask_calls;
        uint32_t a = pick_random_legal(mask, picker);
        if (a == UINT32_MAX) {
            reset_one(s, b, picker.next()); ++reset_calls;
            continue;
        }
        float r; uint8_t d;
        step_one(s, b, a, r, d);
        if (d) { reset_one(s, b, picker.next()); ++reset_calls; ++game_count; }
    }

    auto t0 = clk::now();
    while (step_count < TOTAL_STEPS) {
        compute_mask(s, b, mask);
        ++mask_calls;

        uint32_t a = pick_random_legal(mask, picker);
        if (a == UINT32_MAX) {
            reset_one(s, b, picker.next()); ++reset_calls;
            continue;
        }

        float reward; uint8_t done;
        step_one(s, b, a, reward, done);
        ++step_count;

        if (done) {
            reset_one(s, b, picker.next());
            ++reset_calls;
            ++game_count;
        }
    }
    auto t1 = clk::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    double steps_per_sec = double(step_count) / secs;
    double games_per_sec = (game_count > 0) ? double(game_count) / secs : 0.0;

    std::printf("steps:        %llu\n", (unsigned long long)step_count);
    std::printf("games:        %llu\n", (unsigned long long)game_count);
    std::printf("mask calls:   %llu\n", (unsigned long long)mask_calls);
    std::printf("resets:       %llu\n", (unsigned long long)reset_calls);
    std::printf("elapsed:      %.3f s\n", secs);
    std::printf("steps/sec:    %.0f\n", steps_per_sec);
    if (games_per_sec > 0) std::printf("games/sec:    %.0f\n", games_per_sec);
    std::printf("ns/step:      %.1f\n", secs * 1e9 / double(step_count));
    if (game_count > 0)
        std::printf("steps/game:   %.1f\n", double(step_count) / double(game_count));
    return 0;
}
