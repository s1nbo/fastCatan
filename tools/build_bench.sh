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
