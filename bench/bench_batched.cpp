// Batched throughput benchmark.
// Random-legal-action loop inside C++, no Python overhead.
//
// Build:
//   bash tools/build_bench.sh
// Run:
//   build/bench_batched [n_envs] [n_step_passes]
#include "batched_env.hpp"
#include "rules.hpp"
#include "mask.hpp"
#include "rng.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <vector>

using namespace catan;
using clk = std::chrono::high_resolution_clock;

static inline uint32_t pick_random_legal(const uint64_t* m, Xoshiro128& r) noexcept {
    uint32_t total = 0;
    for (uint32_t w = 0; w < MASK_WORDS; ++w) total += uint32_t(__builtin_popcountll(m[w]));
    if (total == 0) return UINT32_MAX;
    uint32_t pick = r.bounded(total);
    for (uint32_t w = 0; w < MASK_WORDS; ++w) {
        uint32_t pc = uint32_t(__builtin_popcountll(m[w]));
        if (pick < pc) {
            uint64_t v = m[w];
            for (uint32_t i = 0; i < pick; ++i) v &= v - 1;
            return w * 64 + uint32_t(__builtin_ctzll(v));
        }
        pick -= pc;
    }
    return UINT32_MAX;
}

int main(int argc, char** argv) {
    uint32_t n_envs       = (argc > 1) ? uint32_t(std::strtoul(argv[1], nullptr, 10)) : 1024;
    uint32_t n_step_passes = (argc > 2) ? uint32_t(std::strtoul(argv[2], nullptr, 10)) : 2000;

    BatchedEnv env{};
    batched_env_init(env, n_envs, 0xCAFEBABEULL);
    batched_env_reset(env);

    std::vector<uint64_t> masks(std::size_t(n_envs) * MASK_WORDS);
    std::vector<uint32_t> actions(n_envs);
    std::vector<float>    rewards(n_envs);
    std::vector<uint8_t>  dones(n_envs);

    Xoshiro128 picker;
    xoshiro_seed(picker, 0xBEEFFACEULL);

    // Warmup
    for (uint32_t pass = 0; pass < 200; ++pass) {
        batched_env_write_masks(env, masks.data());
        for (uint32_t i = 0; i < n_envs; ++i) {
            uint32_t a = pick_random_legal(masks.data() + i * MASK_WORDS, picker);
            actions[i] = (a == UINT32_MAX) ? 0 : a;
        }
        batched_env_step(env, actions.data(), rewards.data(), dones.data());
    }

    auto t0 = clk::now();
    uint64_t done_count = 0;
    for (uint32_t pass = 0; pass < n_step_passes; ++pass) {
        batched_env_write_masks(env, masks.data());
        for (uint32_t i = 0; i < n_envs; ++i) {
            uint32_t a = pick_random_legal(masks.data() + i * MASK_WORDS, picker);
            actions[i] = (a == UINT32_MAX) ? 0 : a;
        }
        batched_env_step(env, actions.data(), rewards.data(), dones.data());
        for (uint32_t i = 0; i < n_envs; ++i) if (dones[i]) ++done_count;
    }
    auto t1 = clk::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    uint64_t total_steps = uint64_t(n_envs) * n_step_passes;

    std::printf("n_envs:       %u\n",   n_envs);
    std::printf("step passes:  %u\n",   n_step_passes);
    std::printf("total steps:  %llu\n", (unsigned long long)total_steps);
    std::printf("games done:   %llu\n", (unsigned long long)done_count);
    std::printf("elapsed:      %.3f s\n", secs);
    std::printf("steps/sec:    %.0f\n", double(total_steps) / secs);
    std::printf("ns/step:      %.1f\n", secs * 1e9 / double(total_steps));
    if (done_count > 0)
        std::printf("games/sec:    %.0f\n", double(done_count) / secs);

    batched_env_destroy(env);
    return 0;
}
