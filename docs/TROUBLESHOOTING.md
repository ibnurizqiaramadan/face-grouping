# Troubleshooting Guide

## Common Errors and Solutions

### 1. GPU Out of Memory (OOM)

#### Error Message:
```
onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: 
[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION
CUDA failure 2: out of memory
```

#### Cause:
Too many extraction processes running simultaneously on the same GPU.

#### Solution:
Reduce `NUM_EXTRACTION_PROCESSES` in `src/app_dual_gpu.py`:

```python
NUM_EXTRACTION_PROCESSES = 2  # Start with 2 (1 per GPU)
```

#### Guidelines:
- **8GB GPU**: Maximum 2-4 total processes (1-2 per GPU)
- **12GB GPU**: Maximum 4-6 total processes (2-3 per GPU)
- **16GB+ GPU**: Maximum 6-8 total processes (3-4 per GPU)

**Formula**: `Max processes = (GPU_Memory_GB / 4) per GPU`

---

### 2. Process Failed (KeyError)

#### Error Message:
```
KeyError: 0
File "src/app_dual_gpu.py", line 192, in main
    results = [result_dict[i] for i in range(NUM_EXTRACTION_PROCESSES)]
```

#### Cause:
One or more extraction processes crashed before completing (usually due to GPU OOM).

#### Solution:
The script now handles this gracefully. You'll see:
```
⚠️  Process 0 failed (likely GPU OOM)
💡 Try reducing NUM_EXTRACTION_PROCESSES
```

**Action**: Reduce `NUM_EXTRACTION_PROCESSES` and retry.

---

### 3. Slow Extraction Speed

#### Symptom:
Extraction speed < 2 images/second per process.

#### Possible Causes:
1. GPU not fully utilized
2. Too few processes
3. Disk I/O bottleneck

#### Solutions:

**A. Check GPU Utilization:**
```bash
./monitor_gpu.sh
```

If GPU utilization < 80%, increase processes:
```python
NUM_EXTRACTION_PROCESSES = 6  # Carefully increase
```

**B. Check Disk Speed:**
```bash
# Test read speed
dd if=/mnt/data/face-grouping/dataset/sample.jpg of=/dev/null bs=1M count=100
```

If slow (< 50 MB/s), consider:
- Moving dataset to SSD
- Using NVMe storage
- Reducing `NUM_DRAWING_WORKERS`

---

### 4. Clustering Takes Too Long

#### Symptom:
Clustering > 5 seconds for < 100 images.

#### Solution:
Increase clustering workers:
```python
NUM_CLUSTERING_WORKERS = 16  # Increase for large datasets
```

**Note**: This is CPU-bound, so more workers = faster (up to your CPU core count).

---

### 5. Drawing Too Slow

#### Symptom:
Drawing speed < 5 images/second.

#### Possible Causes:
1. HDD storage (slow disk I/O)
2. Too few workers
3. CPU bottleneck

#### Solutions:

**A. Increase Workers:**
```python
NUM_DRAWING_WORKERS = 16  # More workers for better parallelism
```

**B. Use Faster Storage:**
- Move output folder to SSD
- Use tmpfs for temporary storage

**C. Check CPU Usage:**
```bash
htop
```

If all cores maxed out, you've reached CPU limit.

---

### 6. Same Person Split Across Multiple Clusters

#### Symptom:
Same person appears as person_0 and person_5.

#### Cause:
`THRESHOLD` too low (too strict matching).

#### Solution:
Increase threshold:
```python
THRESHOLD = 1.2  # Looser matching
```

**Range**: 0.8 (strict) to 1.5 (very loose)

---

### 7. Different People in Same Cluster

#### Symptom:
Different people grouped together incorrectly.

#### Cause:
`THRESHOLD` too high (too loose matching).

#### Solution:
Decrease threshold:
```python
THRESHOLD = 0.8  # Stricter matching
```

---

### 8. Redis Connection Failed

#### Error Message:
```
⚠️  Redis not available: Connection refused
```

#### Cause:
Redis server not running.

#### Solution:

**A. Start Redis:**
```bash
sudo systemctl start redis
```

**B. Check Redis Status:**
```bash
redis-cli ping
# Should return: PONG
```

**C. Install Redis (if not installed):**
```bash
sudo apt install redis-server -y
sudo systemctl enable redis
sudo systemctl start redis
```

**Note**: Script will continue without Redis, but results won't be persisted.

---

### 9. CUDA Not Available

#### Error Message:
```
Applied providers: ['CPUExecutionProvider']
```

(No `CUDAExecutionProvider`)

#### Cause:
ONNX Runtime can't find CUDA libraries.

#### Solution:

**A. Check CUDA Installation:**
```bash
nvidia-smi
nvcc --version
```

**B. Reinstall CUDA libraries:**
```bash
pip install --force-reinstall onnxruntime-gpu==1.17.1
```

**C. Set Library Path:**
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

**D. Use the provided script:**
```bash
./scripts/run_dual_gpu.sh  # Already sets LD_LIBRARY_PATH
```

---

### 10. Progress Bars Overlapping

#### Symptom:
Multiple progress bars printing on top of each other, hard to read.

#### Cause:
Multiple processes/workers writing to stdout simultaneously.

#### Solution:
This is cosmetic only. The script is working correctly.

To reduce clutter, redirect output:
```bash
./scripts/run_dual_gpu.sh 2>&1 | tee output.log
```

---

## Performance Optimization Tips

### 1. Find Your Optimal Configuration

**Step 1**: Start conservative
```python
NUM_EXTRACTION_PROCESSES = 2
NUM_CLUSTERING_WORKERS = 4
NUM_DRAWING_WORKERS = 4
```

**Step 2**: Test with small dataset (20 images)

**Step 3**: Monitor GPU memory:
```bash
watch -n 0.5 nvidia-smi
```

**Step 4**: Gradually increase if memory allows:
```python
NUM_EXTRACTION_PROCESSES = 4  # If GPU memory < 80%
```

**Step 5**: Test full dataset

---

### 2. Benchmark Your System

Run with different configs and note times:

```bash
# Test 1: Conservative
NUM_EXTRACTION_PROCESSES=2 ./scripts/run_dual_gpu.sh

# Test 2: Balanced
NUM_EXTRACTION_PROCESSES=4 ./scripts/run_dual_gpu.sh

# Test 3: Aggressive (may OOM)
NUM_EXTRACTION_PROCESSES=6 ./scripts/run_dual_gpu.sh
```

Compare total times and find sweet spot.

---

### 3. Hardware-Specific Recommendations

#### Tesla P4 (8GB VRAM)
```python
NUM_EXTRACTION_PROCESSES = 2  # 1 per GPU
NUM_CLUSTERING_WORKERS = 8
NUM_DRAWING_WORKERS = 8
```

#### Tesla T4 (16GB VRAM)
```python
NUM_EXTRACTION_PROCESSES = 4  # 2 per GPU
NUM_CLUSTERING_WORKERS = 8
NUM_DRAWING_WORKERS = 8
```

#### RTX 3090 (24GB VRAM)
```python
NUM_EXTRACTION_PROCESSES = 6  # 3 per GPU
NUM_CLUSTERING_WORKERS = 16
NUM_DRAWING_WORKERS = 16
```

#### A100 (40GB+ VRAM)
```python
NUM_EXTRACTION_PROCESSES = 8  # 4 per GPU
NUM_CLUSTERING_WORKERS = 16
NUM_DRAWING_WORKERS = 16
```

---

## Debug Mode

### Enable Verbose Logging

Add to `src/app_dual_gpu.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check GPU Memory Usage

```bash
# Before running
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# During execution (separate terminal)
watch -n 1 nvidia-smi
```

### Profile Performance

```bash
# Time each stage
time ./scripts/run_dual_gpu.sh

# Or use Python profiler
python -m cProfile -o profile.stats src/app_dual_gpu.py
```

---

## Getting Help

If issues persist:

1. **Check logs**: `./scripts/run_dual_gpu.sh 2>&1 | tee debug.log`
2. **Check GPU memory**: `nvidia-smi`
3. **Check disk space**: `df -h`
4. **Check system resources**: `htop`
5. **Reduce all workers to minimum** and test

---

**Last Updated**: 2025-10-09  
**Version**: 3.1
