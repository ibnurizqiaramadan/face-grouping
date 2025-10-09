# Configuration Guide

## Performance Configuration

All performance settings are located at the top of `src/app_dual_gpu.py` (lines 25-45).

### Quick Reference

```python
# === PERFORMANCE CONFIG ===
NUM_EXTRACTION_PROCESSES = 4    # Face extraction processes
NUM_CLUSTERING_WORKERS = 8      # Clustering workers
NUM_DRAWING_WORKERS = 8         # Drawing workers
THRESHOLD = 1.0                 # Face similarity threshold
```

---

## 1. NUM_EXTRACTION_PROCESSES

**What it does**: Controls how many parallel processes extract faces from images.

**Default**: `4` (2 processes per GPU)

**How to adjust**:
```python
NUM_EXTRACTION_PROCESSES = 4    # For 2 GPUs, this gives 2 processes per GPU
NUM_EXTRACTION_PROCESSES = 6    # For 2 GPUs, this gives 3 processes per GPU
NUM_EXTRACTION_PROCESSES = 8    # For 2 GPUs, this gives 4 processes per GPU
```

**Guidelines**:
- Must be an even number (will be split between 2 GPUs)
- More processes = faster extraction BUT more GPU memory usage
- **Recommended**: 2-4 processes per GPU
- **Memory**: ~2GB GPU memory per process

**When to increase**:
- Large datasets (1000+ images)
- If GPUs have high memory (16GB+)
- If GPU utilization is < 80%

**When to decrease**:
- If you get "Out of Memory" errors
- If GPUs have limited memory (< 8GB)

---

## 2. NUM_CLUSTERING_WORKERS

**What it does**: Controls parallel workers for finding similar face pairs.

**Default**: `8`

**How to adjust**:
```python
NUM_CLUSTERING_WORKERS = 4     # Conservative (4 CPU cores)
NUM_CLUSTERING_WORKERS = 8     # Balanced (8 CPU cores)
NUM_CLUSTERING_WORKERS = 16    # Aggressive (16 CPU cores)
```

**Guidelines**:
- CPU-bound task (uses CPU, not GPU)
- More workers = faster clustering
- **Recommended**: Match your CPU core count (4-16)
- **Memory**: Minimal impact

**When to increase**:
- Large datasets (500+ images)
- Many CPU cores available
- Clustering takes > 2 seconds

**When to decrease**:
- Limited CPU cores (< 8)
- Other processes need CPU resources

---

## 3. NUM_DRAWING_WORKERS

**What it does**: Controls parallel workers for drawing bounding boxes on images.

**Default**: `8`

**How to adjust**:
```python
NUM_DRAWING_WORKERS = 4     # Conservative
NUM_DRAWING_WORKERS = 8     # Balanced
NUM_DRAWING_WORKERS = 16    # Aggressive
```

**Guidelines**:
- I/O-bound task (disk read/write heavy)
- More workers = faster drawing (with diminishing returns)
- **Recommended**: 4-16 workers
- **Memory**: Low impact

**When to increase**:
- Large datasets (500+ images)
- Fast SSD storage
- Drawing takes > 10 seconds

**When to decrease**:
- Slow disk (HDD)
- Limited CPU cores
- Already fast enough (< 5 seconds)

---

## 4. THRESHOLD

**What it does**: Controls how similar faces must be to be grouped together.

**Default**: `1.0`

**How to adjust**:
```python
THRESHOLD = 0.8    # Strict - same person must look very similar
THRESHOLD = 1.0    # Balanced - allows some variation (lighting, angle)
THRESHOLD = 1.2    # Loose - groups faces even with significant variation
```

**Guidelines**:
- Uses L2 distance on normalized face embeddings
- **Recommended range**: 0.8 - 1.2
- Lower = stricter, Higher = looser

**When to increase**:
- Same person appears in multiple clusters
- Dataset has varied lighting/angles
- Many small clusters (< 3 images)

**When to decrease**:
- Different people grouped together
- Need very strict matching
- Dataset has consistent conditions

---

## Optimization Examples

### Small Dataset (< 100 images)

```python
NUM_EXTRACTION_PROCESSES = 2    # Light load
NUM_CLUSTERING_WORKERS = 4      # Sufficient
NUM_DRAWING_WORKERS = 4         # Sufficient
THRESHOLD = 1.0
```

**Expected**: ~15-20 seconds total

---

### Medium Dataset (100-500 images)

```python
NUM_EXTRACTION_PROCESSES = 4    # Balanced
NUM_CLUSTERING_WORKERS = 8      # Good parallelism
NUM_DRAWING_WORKERS = 8         # Good parallelism
THRESHOLD = 1.0
```

**Expected**: ~60-100 seconds total

---

### Large Dataset (500-2000 images)

```python
NUM_EXTRACTION_PROCESSES = 6    # High parallelism
NUM_CLUSTERING_WORKERS = 16     # Max parallelism
NUM_DRAWING_WORKERS = 16        # Max parallelism
THRESHOLD = 1.0
```

**Expected**: ~200-400 seconds total

---

### Very Large Dataset (2000+ images)

```python
NUM_EXTRACTION_PROCESSES = 8    # Maximum parallelism
NUM_CLUSTERING_WORKERS = 16     # Maximum parallelism
NUM_DRAWING_WORKERS = 16        # Maximum parallelism
THRESHOLD = 1.0
```

**Expected**: 400+ seconds (linear scaling)

---

## Hardware Recommendations

### Minimum Requirements
- **GPU**: 1x GPU with 8GB VRAM
- **CPU**: 4 cores
- **RAM**: 16GB
- **Storage**: HDD

**Config**:
```python
NUM_EXTRACTION_PROCESSES = 2
NUM_CLUSTERING_WORKERS = 4
NUM_DRAWING_WORKERS = 4
```

---

### Recommended Setup
- **GPU**: 2x GPU with 8-12GB VRAM each
- **CPU**: 8-16 cores
- **RAM**: 32GB
- **Storage**: SSD

**Config**:
```python
NUM_EXTRACTION_PROCESSES = 4
NUM_CLUSTERING_WORKERS = 8
NUM_DRAWING_WORKERS = 8
```

---

### High Performance Setup
- **GPU**: 2x GPU with 16GB+ VRAM each
- **CPU**: 16-32 cores
- **RAM**: 64GB+
- **Storage**: NVMe SSD

**Config**:
```python
NUM_EXTRACTION_PROCESSES = 8
NUM_CLUSTERING_WORKERS = 16
NUM_DRAWING_WORKERS = 16
```

---

## Troubleshooting

### "Out of Memory" Error
**Solution**: Reduce `NUM_EXTRACTION_PROCESSES`
```python
NUM_EXTRACTION_PROCESSES = 2  # Use fewer processes
```

---

### Slow Extraction
**Solution**: Increase `NUM_EXTRACTION_PROCESSES` if GPU memory allows
```python
NUM_EXTRACTION_PROCESSES = 6  # More processes
```

---

### Slow Clustering
**Solution**: Increase `NUM_CLUSTERING_WORKERS`
```python
NUM_CLUSTERING_WORKERS = 16  # More workers
```

---

### Slow Drawing
**Solution**: Increase `NUM_DRAWING_WORKERS`
```python
NUM_DRAWING_WORKERS = 16  # More workers
```

---

### Same Person in Multiple Clusters
**Solution**: Increase `THRESHOLD`
```python
THRESHOLD = 1.2  # Looser matching
```

---

### Different People in Same Cluster
**Solution**: Decrease `THRESHOLD`
```python
THRESHOLD = 0.8  # Stricter matching
```

---

## Testing Your Configuration

After making changes, test with a small subset:

1. Copy 20-30 images to a test folder
2. Update `DATA_DIR` temporarily
3. Run: `./scripts/run_dual_gpu.sh`
4. Check results in `clusters_output_labeled/`
5. Adjust THRESHOLD if needed
6. Restore `DATA_DIR` and run full dataset

---

## Monitoring Performance

### Watch GPU Usage
```bash
./scripts/monitor_gpu.sh
```

Look for:
- **GPU Utilization**: Should be 80-100% during extraction
- **GPU Memory**: Should be < 80% of total
- **Temperature**: Should be < 80°C

### Check Results
```bash
python src/check_redis.py
```

Look for:
- **Total clusters**: Reasonable for your dataset
- **Images per cluster**: Not too many or too few
- **Processing time**: Acceptable for your needs

---

## Advanced Tips

1. **Balance is key**: More workers isn't always better
2. **Monitor first**: Check GPU/CPU usage before adjusting
3. **Test incrementally**: Change one value at a time
4. **SSD helps**: Drawing speed improves significantly with SSD
5. **Memory matters**: More GPU memory = more extraction processes

---

**Last Updated**: 2025-10-09  
**Version**: 3.1
