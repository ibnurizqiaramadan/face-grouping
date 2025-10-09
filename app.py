import os
import cv2
import numpy as np
import faiss
from tqdm import tqdm
from insightface.app import FaceAnalysis
from collections import defaultdict
import shutil
import redis
import json
from datetime import datetime

# === SETUP ===
DATA_DIR = "/mnt/data/face-grouping/dataset"   # folder berisi semua foto kamu
OUTPUT_DIR_LABELED = "clusters_output_labeled"  # Folder untuk foto dengan box
os.makedirs(OUTPUT_DIR_LABELED, exist_ok=True)

# === REDIS CONFIG ===
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

# Connect to Redis
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client.ping()
    print(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    print(f"⚠️  Redis not available: {e}")
    redis_client = None

# === INIT MODEL ===
app = FaceAnalysis(name='buffalo_l')  # ArcFace + RetinaFace
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 = GPU

# === LOAD DAN EKSTRAK EMBEDDING ===
embeddings = []
paths = []
face_boxes = []  # Simpan bounding boxes

print("🔍 Extracting face embeddings...")
for file in tqdm(os.listdir(DATA_DIR)):
    img_path = os.path.join(DATA_DIR, file)
    img = cv2.imread(img_path)
    if img is None:
        continue
    faces = app.get(img)
    if len(faces) == 0:
        continue
    emb = faces[0].normed_embedding
    bbox = faces[0].bbox.astype(int)  # [x1, y1, x2, y2]
    embeddings.append(emb)
    paths.append(img_path)
    face_boxes.append(bbox)

embeddings = np.array(embeddings).astype('float32')

# === CLUSTERING DENGAN FAISS ===
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# threshold kemiripan wajah (semakin BESAR, semakin loose)
# Untuk cosine similarity, nilai bagus: 0.4-0.6
# Untuk L2 distance dengan normalized embeddings: 0.8-1.2
THRESHOLD = 1.0  # Dinaikkan dari 0.6 untuk menangkap variasi pose/lighting

print(f"\n🔬 Clustering dengan threshold={THRESHOLD}...")

# Build similarity graph menggunakan Union-Find (untuk transitive clustering)
# Jika A mirip B, dan B mirip C, maka A, B, C dalam satu cluster
parent = list(range(len(embeddings)))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[px] = py

# Bandingkan semua pairs
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        dist = np.linalg.norm(embeddings[i] - embeddings[j])
        if dist < THRESHOLD:
            union(i, j)

# Kumpulkan clusters
cluster_map = defaultdict(list)
for i in range(len(embeddings)):
    cluster_map[find(i)].append(i)

clusters = list(cluster_map.values())

# === SIMPAN HASIL ===
# Buat mapping dari index ke cluster ID
idx_to_cluster = {}
for cluster_id, cluster in enumerate(clusters):
    for i in cluster:
        idx_to_cluster[i] = cluster_id

print("\n💾 Saving results...")

# Simpan foto dengan bounding box dan label
print("🎨 Drawing bounding boxes on images...")
for i in tqdm(range(len(paths))):
    img_path = paths[i]
    img = cv2.imread(img_path)
    bbox = face_boxes[i]
    cluster_id = idx_to_cluster[i]
    
    # Gambar bounding box
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0)  # Hijau
    thickness = 3
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Tambahkan label
    label = f"person_{cluster_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    
    # Background untuk text
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
    
    # Tulis text
    cv2.putText(img, label, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)
    
    # Simpan ke folder labeled
    cluster_dir_labeled = os.path.join(OUTPUT_DIR_LABELED, f"person_{cluster_id}")
    os.makedirs(cluster_dir_labeled, exist_ok=True)
    filename = os.path.basename(img_path)
    output_path = os.path.join(cluster_dir_labeled, filename)
    cv2.imwrite(output_path, img)

# === SIMPAN KE REDIS ===
if redis_client:
    print("\n📤 Saving clustering results to Redis...")
    
    # Prepare data untuk Redis
    clustering_result = {
        "timestamp": datetime.now().isoformat(),
        "total_images": len(paths),
        "total_clusters": len(clusters),
        "threshold": THRESHOLD,
        "clusters": []
    }
    
    for cluster_id, cluster in enumerate(clusters):
        cluster_data = {
            "cluster_id": cluster_id,
            "person_label": f"person_{cluster_id}",
            "image_count": len(cluster),
            "images": []
        }
        
        for i in cluster:
            img_info = {
                "filename": os.path.basename(paths[i]),
                "path": paths[i],
                "bbox": face_boxes[i].tolist(),
                "labeled_output": os.path.join(OUTPUT_DIR_LABELED, f"person_{cluster_id}", os.path.basename(paths[i]))
            }
            cluster_data["images"].append(img_info)
        
        clustering_result["clusters"].append(cluster_data)
    
    # Simpan ke Redis dengan key: face_clustering:latest
    redis_key = "face_clustering:latest"
    redis_client.set(redis_key, json.dumps(clustering_result))
    
    # Simpan juga dengan timestamp sebagai key untuk history
    history_key = f"face_clustering:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    redis_client.set(history_key, json.dumps(clustering_result))
    
    # Set expiry untuk history keys (7 hari)
    redis_client.expire(history_key, 7 * 24 * 60 * 60)
    
    print(f"   ✅ Saved to Redis key: {redis_key}")
    print(f"   ✅ Saved to Redis key: {history_key} (expires in 7 days)")

print(f"\n✅ Clustering selesai. Total {len(clusters)} orang ditemukan.")
print(f"📁 Hasil dengan box & label: {OUTPUT_DIR_LABELED}")
