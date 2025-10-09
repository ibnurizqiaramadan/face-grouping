# Scripts Directory

This folder contains utility scripts for running the face clustering application.

## Available Scripts

### 1. run_dual_gpu.sh
**Main script for production use**

Runs the dual GPU face clustering with optimal performance.

```bash
./scripts/run_dual_gpu.sh
```

**Features:**
- Automatically sets CUDA library paths
- Uses 2 GPUs in parallel
- Activates conda environment
- Runs `src/app_dual_gpu.py`

**Performance:**
- ~2.5 images/sec throughput
- 4 extraction processes (2 per GPU)
- 8 clustering workers
- 8 drawing workers

---

### 2. run_gpu.sh
**Single GPU version**

Runs face clustering on a single GPU (simpler, slower).

```bash
./scripts/run_gpu.sh
```

**Features:**
- Automatically sets CUDA library paths
- Uses 1 GPU
- Activates conda environment
- Runs `src/app.py`

**Performance:**
- ~1.3 images/sec throughput
- Single process extraction
- Good for small datasets (< 100 images)

---

### 3. monitor_gpu.sh
**GPU monitoring utility**

Monitors GPU usage in real-time.

```bash
./scripts/monitor_gpu.sh
```

**Shows:**
- GPU utilization percentage
- Memory usage (used/total)
- Running processes
- Temperature

**Usage:**
Run in a separate terminal while clustering is running:
```bash
# Terminal 1
./scripts/run_dual_gpu.sh

# Terminal 2
./scripts/monitor_gpu.sh
```

---

## Usage Examples

### Basic Usage
```bash
# Recommended: Dual GPU
./scripts/run_dual_gpu.sh
```

### With Monitoring
```bash
# Terminal 1: Run clustering
./scripts/run_dual_gpu.sh

# Terminal 2: Monitor GPUs
watch -n 1 ./scripts/monitor_gpu.sh
```

### Custom Configuration
```bash
# Override environment variables
export REDIS_HOST=192.168.1.100
export REDIS_PORT=6380
./scripts/run_dual_gpu.sh
```

### Debug Mode
```bash
# Redirect output to log file
./scripts/run_dual_gpu.sh 2>&1 | tee clustering_$(date +%Y%m%d_%H%M%S).log
```

---

## Requirements

All scripts require:
- Conda environment: `facecluster-gpu`
- CUDA libraries properly installed
- Redis server running (for dual GPU script)
- Dataset in `/mnt/data/face-grouping/dataset`

---

## Script Internals

### Environment Setup
All scripts automatically:
1. Activate conda environment
2. Set `LD_LIBRARY_PATH` for CUDA libraries
3. Configure GPU visibility

### CUDA Library Paths
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
```

---

## Troubleshooting

### Script Permission Denied
```bash
chmod +x scripts/*.sh
```

### CUDA Libraries Not Found
Use the scripts instead of running Python directly:
```bash
# ❌ May fail
python src/app_dual_gpu.py

# ✅ Always works
./scripts/run_dual_gpu.sh
```

### Conda Environment Not Found
```bash
# Create environment first
conda env create -f environment.yml
# Or
conda create -n facecluster-gpu python=3.12
```

---

**Last Updated**: 2025-10-09  
**Version**: 3.1
