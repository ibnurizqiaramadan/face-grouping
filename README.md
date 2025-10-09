# Face Clustering dengan GPU

Aplikasi untuk mengelompokkan foto berdasarkan wajah menggunakan **InsightFace + FAISS GPU**.

## Setup Environment

### 1. Buat dan aktifkan conda environment
```bash
conda create -n facecluster-gpu python=3.12 -y
conda activate facecluster-gpu
```

### 2. Install FAISS-GPU (via conda)
```bash
conda install -c pytorch -c nvidia faiss-gpu pytorch-cuda=11.8 -y
```

### 3. Install dependencies lainnya (via pip)
```bash
pip install nvidia-pyindex
pip install -r requirements.txt
```

## Cara Pakai

### Option 1: Single GPU (Standard)
```bash
./run_gpu.sh
```

### Option 2: Dual GPU (Parallel Processing - FASTER!)
```bash
./run_dual_gpu.sh
```

**Dual GPU Mode Features:**
- Split dataset secara otomatis ke 2 GPU
- Process parallel dengan multiprocessing
- ~10% lebih cepat untuk dataset besar
- Merge hasil otomatis sebelum clustering

### Option 3: Jalankan langsung
```bash
conda activate facecluster-gpu
python app.py              # Single GPU
# atau
python app_dual_gpu.py     # Dual GPU
```

**Note:** Jika langsung pakai `python` dan dapat error CUDA libraries, gunakan script `run_*.sh` yang sudah set LD_LIBRARY_PATH otomatis.

## Konfigurasi

### Environment Variables (Redis)
```bash
export REDIS_HOST=localhost  # Default: localhost
export REDIS_PORT=6379       # Default: 6379
export REDIS_DB=0            # Default: 0
```

### Edit `app.py`
- `DATA_DIR`: Folder input foto
- `OUTPUT_DIR_LABELED`: Folder output hasil clustering dengan label
- `THRESHOLD`: Tingkat kemiripan wajah (default: 1.0, semakin besar = semakin loose)

## Performance

### End-to-End Pipeline (78 images)

| Mode | Extraction | Clustering | Drawing | Total | Overall Speed |
|------|-----------|-----------|---------|-------|---------------|
| CPU | ~56s | ~2s | ~42s | ~98s | 0.8 img/s |
| Single GPU | ~30s | ~2s | ~42s | ~72s | 1.08 img/s |
| Dual GPU (2 workers) | ~25s | ~1s | ~22s | ~47s | 1.66 img/s |
| **Dual GPU (8 workers)** | **~22s** | **~1s** | **~8s** | **~32s** | **2.45 img/s** 🚀 |

### Speedup Analysis
- **vs CPU**: +207% faster (98s → 32s) 🔥
- **vs Single GPU**: +132% faster (72s → 32s) ⚡  
- **vs Dual GPU (2 draw workers)**: +48% faster (47s → 32s)

### Key Features
- ✅ Parallel face extraction pada 2 GPU (3.48 img/s)
- ✅ GPU clustering dengan 8 workers (~1s)
- ✅ Parallel bounding box drawing 8 workers (9.42 img/s) 🔥
- ✅ Overall throughput: **2.45 images/sec**

📊 **See [PERFORMANCE.md](PERFORMANCE.md) for detailed benchmarks**

## Output

### 1. Foto dengan Bounding Box & Label
```
clusters_output_labeled/
├── person_0/
│   ├── IMG_001.jpg  (dengan green box + label "person_0")
│   └── IMG_002.jpg  (dengan green box + label "person_0")
├── person_1/
│   ├── IMG_003.jpg  (dengan green box + label "person_1")
│   └── IMG_004.jpg  (dengan green box + label "person_1")
...
```

Setiap folder berisi foto dari orang yang sama dengan bounding box hijau dan label person ID di setiap wajah.

### 2. Redis Storage
Hasil clustering juga disimpan di Redis dengan struktur:

**Key: `face_clustering:latest`** (hasil clustering terbaru)
```json
{
  "timestamp": "2025-10-09T00:33:00",
  "total_images": 78,
  "total_clusters": 17,
  "threshold": 1.0,
  "clusters": [
    {
      "cluster_id": 0,
      "person_label": "person_0",
      "image_count": 2,
      "images": [
        {
          "filename": "IMG_001.jpg",
          "path": "/path/to/original/IMG_001.jpg",
          "bbox": [x1, y1, x2, y2],
          "labeled_output": "/path/to/output/person_0/IMG_001.jpg"
        }
      ]
    }
  ]
}
```

**Key: `face_clustering:YYYYMMDD_HHMMSS`** (history dengan timestamp, expire setelah 7 hari)

### Cara Baca dari Redis
```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
result = json.loads(r.get('face_clustering:latest'))
print(f"Total clusters: {result['total_clusters']}")
```
