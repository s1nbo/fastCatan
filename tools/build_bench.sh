#!/usr/bin/env bash
# Build the throughput benchmark.
# Output: build/bench_step
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build

clang++ -std=c++23 -O3 -march=native -DNDEBUG -Wall -Wextra \
    -I include \
    src/catan/rules.cpp bench/bench_step.cpp \
    -o build/bench_step
echo "built build/bench_step"

clang++ -std=c++23 -O3 -march=native -DNDEBUG -Wall -Wextra \
    -I include \
    src/catan/rules.cpp src/catan/batched_env.cpp src/catan/obs.cpp \
    bench/bench_batched.cpp \
    -o build/bench_batched
echo "built build/bench_batched"
