# Running fastCatan on HPC

Quick reference for building and training fastCatan on a Linux HPC cluster
(GCC 14.2 + CUDA + module-based environment).

## 1. Module setup

```bash
# Adjust module names to your cluster.
module purge
module load gcc/14.2          # C++23 + OpenMP
module load cmake/3.27        # required minimum
module load python/3.12       # for venv
module load cuda/12.4         # if training on GPU
```

Verify versions:

```bash
gcc --version           # >= 14.0
cmake --version         # >= 3.27
python3 --version       # >= 3.10
nvidia-smi              # if using CUDA
```

## 2. Clone + create venv

```bash
git clone <repo> fastcatan
cd fastcatan

python3 -m venv .venv
source .venv/bin/activate

# Pinned CMake/ninja so build doesn't pick up the system ones.
pip install --upgrade pip
pip install cmake ninja nanobind scikit-build-core
```

## 3. Build + install fastcatan (editable)

```bash
pip install -e . --no-build-isolation
```

This will:

- Compile `src/catan/*.cpp` with `-O3 -march=native -fno-exceptions -fno-rtti` + LTO.
- Auto-detect OpenMP and link it (the batched env will parallelize across cores).
- Build the nanobind extension `_fastcatan.cpython-XYZ.so` and install into `.venv`.

Verify:

```bash
python3 -c "import fastcatan as fc; print(fc.OBS_SIZE, fc.NUM_ACTIONS)"
# Should print: 724 296
```

## 4. Build the standalone benchmarks

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

build/bench_step           # single-env throughput
build/bench_batched 4096 5000   # batched (will use OpenMP if available)
```

## 5. Install RL training stack

```bash
pip install torch sb3-contrib stable-baselines3 numpy
```

For CUDA wheels (if `nvidia-smi` shows a GPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then verify GPU access:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 6. Run the smoke trainer

```bash
python3 tools/train_smoke.py --total-timesteps 10000 --device cuda --eval-episodes 50
```

For a real training run, scale up:

```bash
python3 tools/train_smoke.py \
    --total-timesteps 1000000 \
    --device cuda \
    --eval-episodes 200 \
    --save checkpoints/ppo_random_v0.zip
```

## 7. Run all unit tests

```bash
for t in tools/test_*.py; do
    echo "=== $t ===";
    python3 "$t" 2>&1 | tail -3;
done
```

Expected: every suite ends with `ALL TESTS PASS` (or `ALL PERFT HASHES MATCH` for `test_perft.py`).

## 8. Submit a SLURM job

Example single-node job for training:

```bash
#!/bin/bash
#SBATCH --job-name=fastcatan-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out

module load gcc/14.2 cmake python/3.12 cuda/12.4
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python3 tools/train_smoke.py \
    --total-timesteps 5000000 \
    --device cuda \
    --eval-episodes 500 \
    --save checkpoints/ppo_v0_$SLURM_JOB_ID.zip
```

## 9. Common pitfalls

| Symptom | Fix |
|---|---|
| `lipo: can't figure out the architecture type of: /opt/.../cmake` | Install pinned cmake into the venv: `pip install cmake ninja` |
| `nanobind/nb_descr.h: typeid requires -frtti` | The nanobind module CMake target must NOT inherit `-fno-rtti`/`-fno-exceptions`. Already gated in `CMakeLists.txt` — confirm `if(SKBUILD)` block uses fresh flags. |
| `OpenMP not found` at configure | Cluster might have GCC without libgomp. Install via `module load libgomp` or symlink. The build will still work single-threaded. |
| Static-assertion failed at `sizeof(GameState)` | Compiler added struct padding differently than expected. Check `alignof(GameState) == 64` and adjust the `static_assert(sizeof(GameState) == 384)` in `state.hpp`. |
| `import fastcatan` returns empty module | Check `.venv/lib/.../fastcatan/` contains BOTH `_fastcatan*.so` and `__init__.py`. Reinstall with `pip install -e . --no-build-isolation`. |

## 10. Performance expectations

| Run | Throughput |
|---|---|
| `bench_batched 4096` (single core) | ~12M steps/sec |
| `bench_batched 4096` with OMP_NUM_THREADS=32 | ~150-300M steps/sec (depends on memory bandwidth) |
| `train_smoke.py` via SB3 (Python loop dominates) | ~200-500 fps (env steps/sec) |

The native batched path is fast; the Python RL loop is the bottleneck during
training. Future optimization: drive PPO directly off `BatchedEnv` with
torch tensors instead of going through Gymnasium.

## 11. Reproducibility

- Per-env RNG seeded via SplitMix64 from a master seed; identical sequences guaranteed.
- `tools/perft_hashes.json` contains pinned trajectory hashes; `python3 tools/test_perft.py` verifies bit-exact reproducibility.
- Document your cluster's `gcc --version`, `glibc` version, and the seed schedule used in any run.
