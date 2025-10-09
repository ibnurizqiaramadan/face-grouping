# Changelog

## v2.1.0 - Parallel Drawing (2025-10-09)

### ✨ New Features
- **Parallel Bounding Box Drawing**: Draw boxes dengan 2 workers parallel (+91% faster)
- Overall pipeline speedup: **53% faster** vs single GPU

### 📊 Performance (78 images)
- Extraction: 25s (3.09 img/s) - Parallel on 2 GPUs
- Drawing: 22s (3.58 img/s) - Parallel with 2 workers
- **Total: 47s (1.66 img/s overall)**

## v2.0.0 - Dual GPU Support (2025-10-09)

### ✨ New Features
- **Dual GPU Parallel Processing**: Split dataset dan process di 2 GPU secara bersamaan
- **Redis Integration**: Simpan clustering results ke Redis dengan TTL
- **Labeled Output Only**: Hanya simpan foto dengan bounding box & label
- **Improved Clustering Algorithm**: Union-Find untuk transitive clustering

### 📊 Performance
- Single GPU extraction: ~2.6 images/sec
- Dual GPU extraction: ~2.86 images/sec (~10% faster)
- Overall speedup vs CPU: 107%

### 📁 New Files
- `src/app_dual_gpu.py`: Dual GPU version dengan multiprocessing
- `scripts/run_dual_gpu.sh`: Script untuk run dual GPU mode
- `src/check_redis.py`: Helper script untuk baca data dari Redis
- `CHANGELOG.md`: This file

### 🔧 Changes
- Updated clustering algorithm dari greedy ke Union-Find
- THRESHOLD dinaikkan dari 0.6 ke 1.0
- Removed `clusters_output` folder (non-labeled)
- Added Redis client dengan auto-reconnect

### 🐛 Fixes
- Fixed multiprocessing queue blocking issue
- Fixed numpy compatibility (now using <2.0 for faiss-gpu)
- Fixed onnxruntime GPU support dengan proper LD_LIBRARY_PATH

## v1.0.0 - Initial Release

### Features
- Face detection dengan InsightFace
- Face clustering dengan FAISS
- Bounding box visualization
- Single GPU support
