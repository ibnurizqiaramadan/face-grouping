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
from multiprocessing import Process, Manager
import time

# === SETUP ===
DATA_DIR = "/mnt/data/face-grouping/dataset"   # folder berisi semua foto kamu
OUTPUT_DIR_LABELED = "clusters_output_labeled"  # Folder untuk foto dengan box
os.makedirs(OUTPUT_DIR_LABELED, exist_ok=True)

# === REDIS CONFIG ===
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

# === PERFORMANCE CONFIG ===
# Adjust these values based on your hardware capabilities
# Higher values = more parallelism but more memory usage
# ⚠️  WARNING: Increasing NUM_EXTRACTION_PROCESSES may cause GPU OOM (Out Of Memory)!

NUM_EXTRACTION_PROCESSES = 4    # Total processes for face extraction (will be split: 2 per GPU)
                                # Recommended: 2-4 for 8GB GPU, 4-6 for 16GB+ GPU
                                # Memory impact: ~2-3GB GPU memory per process
                                # ⚠️  DO NOT exceed: (GPU_Memory_GB / 4) processes per GPU

NUM_CLUSTERING_WORKERS = 16      # Parallel workers for clustering pair finding
                                # Recommended: 4-16 workers (CPU-bound task)
                                # Higher is better for large datasets

NUM_DRAWING_WORKERS = 16         # Parallel workers for drawing bounding boxes
                                # Recommended: 4-16 workers (I/O-bound task)
                                # Higher is better but has diminishing returns

# === CLUSTERING CONFIG ===
# Threshold untuk kemiripan wajah (semakin BESAR, semakin loose)
# Recommended: 0.8-1.2 for face clustering
# Lower = stricter matching, Higher = looser matching
THRESHOLD = 1.0

def process_images_on_gpu(gpu_id, file_list, result_dict, process_id):
    """
    Process images pada GPU tertentu
    """
    # CRITICAL: Set CUDA_VISIBLE_DEVICES untuk force ke GPU ini
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"🚀 GPU {gpu_id} Process {process_id}: Starting processing {len(file_list)} images...")
    
    # Init model pada GPU ini (akan pakai GPU 0 dari perspective process ini)
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    embeddings = []
    paths = []
    face_boxes = []
    
    for file in tqdm(file_list, desc=f"GPU{gpu_id}P{process_id}", position=process_id):
        img_path = os.path.join(DATA_DIR, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = app.get(img)
        if len(faces) == 0:
            continue
        emb = faces[0].normed_embedding
        bbox = faces[0].bbox.astype(int)
        
        embeddings.append(emb)
        paths.append(img_path)
        face_boxes.append(bbox)
    
    # Simpan hasil ke dict (thread-safe dengan Manager)
    result_dict[process_id] = {
        'gpu_id': gpu_id,
        'process_id': process_id,
        'embeddings': embeddings,
        'paths': paths,
        'face_boxes': face_boxes
    }
    
    print(f"✅ GPU {gpu_id} Process {process_id}: Completed {len(embeddings)} faces")

def draw_boxes_parallel(worker_id, indices, paths, face_boxes, idx_to_cluster):
    """
    Draw bounding boxes secara parallel
    """
    for i in tqdm(indices, desc=f"Worker {worker_id}", position=worker_id):
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
    
    print(f"✅ Worker {worker_id}: Completed drawing {len(indices)} images")

def main():
    print("\n" + "="*60)
    print("🚀 DUAL GPU FACE CLUSTERING")
    print(f"   Extraction: {NUM_EXTRACTION_PROCESSES} processes")
    print(f"   Clustering: {NUM_CLUSTERING_WORKERS} workers")
    print(f"   Drawing: {NUM_DRAWING_WORKERS} workers")
    print("="*60 + "\n")
    
    # Connect to Redis
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        redis_client.ping()
        print(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        redis_client = None
    
    # Ambil semua file
    all_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n📸 Total files: {len(all_files)}")
    
    # Split menjadi N bagian untuk parallel extraction (balanced distribution)
    total_files = len(all_files)
    base_chunk_size = total_files // NUM_EXTRACTION_PROCESSES
    remainder = total_files % NUM_EXTRACTION_PROCESSES
    
    file_chunks = []
    current_idx = 0
    for i in range(NUM_EXTRACTION_PROCESSES):
        # Processes dengan ID < remainder mendapat 1 file extra untuk balance
        chunk_size = base_chunk_size + (1 if i < remainder else 0)
        file_chunks.append(all_files[current_idx:current_idx + chunk_size])
        current_idx += chunk_size
    
    # Display split info
    processes_per_gpu = NUM_EXTRACTION_PROCESSES // 2
    for i in range(NUM_EXTRACTION_PROCESSES):
        gpu_id = i // processes_per_gpu
        proc_num = i % processes_per_gpu
        print(f"📦 GPU {gpu_id} Process {proc_num}: {len(file_chunks[i])} files")
    
    print(f"\n🔍 Extracting face embeddings dengan {NUM_EXTRACTION_PROCESSES} processes ({processes_per_gpu} per GPU)...\n")
    
    # Create shared dict untuk hasil (thread-safe)
    manager = Manager()
    result_dict = manager.dict()
    
    # Buat processes untuk extraction
    processes = []
    
    for process_id in range(NUM_EXTRACTION_PROCESSES):
        gpu_id = process_id // processes_per_gpu
        p = Process(target=process_images_on_gpu, args=(gpu_id, file_chunks[process_id], result_dict, process_id))
        processes.append(p)
    
    # Start all processes
    start_time = time.time()
    for p in processes:
        p.start()
    
    # Wait untuk semua process selesai
    for p in processes:
        p.join()
    
    elapsed = time.time() - start_time
    print(f"\n⚡ Extraction selesai dalam {elapsed:.2f} detik")
    
    # Ambil hasil dari dict dan sort by process_id
    # Handle failed processes gracefully
    results = []
    failed_processes = []
    for i in range(NUM_EXTRACTION_PROCESSES):
        if i in result_dict:
            results.append(result_dict[i])
        else:
            failed_processes.append(i)
            print(f"⚠️  Process {i} failed (likely GPU OOM)")
    
    if failed_processes:
        print(f"\n❌ {len(failed_processes)} process(es) failed: {failed_processes}")
        print(f"💡 Try reducing NUM_EXTRACTION_PROCESSES (currently {NUM_EXTRACTION_PROCESSES})")
        if len(results) == 0:
            print("❌ All processes failed! Exiting...")
            return
    
    # Merge semua hasil
    print(f"\n🔄 Merging results dari {NUM_EXTRACTION_PROCESSES} processes...")
    all_embeddings = []
    all_paths = []
    all_face_boxes = []
    
    for result in results:
        all_embeddings.extend(result['embeddings'])
        all_paths.extend(result['paths'])
        all_face_boxes.extend(result['face_boxes'])
    
    print(f"   Total faces extracted: {len(all_embeddings)}")
    
    if len(all_embeddings) == 0:
        print("❌ Tidak ada wajah yang terdeteksi!")
        return
    
    embeddings = np.array(all_embeddings).astype('float32')
    paths = all_paths
    face_boxes = all_face_boxes
    
    # === CLUSTERING DENGAN FAISS GPU ===
    print(f"\n🔬 Clustering dengan threshold={THRESHOLD} (GPU-accelerated)...")
    clustering_start = time.time()
    
    dim = embeddings.shape[1]
    
    # Create GPU index
    print("   📍 Moving FAISS index to GPU 0...")
    res = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatL2(dim)
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    index.add(embeddings)
    
    # Compute all pairwise distances on GPU
    print("   🔢 Computing pairwise distances on GPU...")
    # Search for all neighbors within threshold (use large k to get all)
    k = min(len(embeddings), len(embeddings))
    distances, indices = index.search(embeddings, k)
    
    # Build similarity graph menggunakan Union-Find dengan parallel workers
    print(f"   🔗 Building similarity graph dengan {NUM_CLUSTERING_WORKERS} workers parallel...")
    
    def find_pairs_worker(worker_id, start_idx, end_idx, distances, indices, threshold, result_list):
        """Find all similar pairs in assigned range"""
        pairs = []
        for i in range(start_idx, end_idx):
            for j_idx in range(k):
                j = indices[i, j_idx]
                dist = distances[i, j_idx]
                if i < j and dist < threshold:  # i < j to avoid duplicates
                    pairs.append((i, j))
        result_list.append((worker_id, pairs))
        print(f"      ✅ Worker {worker_id}: Found {len(pairs)} pairs")
    
    # Split work among clustering workers (balanced distribution)
    total_embeddings = len(embeddings)
    base_chunk_size = total_embeddings // NUM_CLUSTERING_WORKERS
    remainder = total_embeddings % NUM_CLUSTERING_WORKERS
    
    manager = Manager()
    result_list = manager.list()
    processes = []
    
    current_idx = 0
    for worker_id in range(NUM_CLUSTERING_WORKERS):
        # Workers dengan ID < remainder mendapat 1 embedding extra untuk balance
        chunk_size = base_chunk_size + (1 if worker_id < remainder else 0)
        start_idx = current_idx
        end_idx = current_idx + chunk_size
        current_idx = end_idx
        
        p = Process(target=find_pairs_worker, args=(worker_id, start_idx, end_idx, distances, indices, THRESHOLD, result_list))
        processes.append(p)
        p.start()
    
    # Wait for all workers
    for p in processes:
        p.join()
    
    # Merge all pairs using Union-Find
    print("   🔗 Merging pairs with Union-Find...")
    parent = list(range(len(embeddings)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Collect all pairs from workers
    all_pairs = []
    for _, pairs in result_list:
        all_pairs.extend(pairs)
    
    print(f"   📊 Total pairs found: {len(all_pairs)}")
    
    # Apply union operations
    for i, j in tqdm(all_pairs, desc="   Union-Find"):
        union(i, j)
    
    # Kumpulkan clusters
    cluster_map = defaultdict(list)
    for i in range(len(embeddings)):
        cluster_map[find(i)].append(i)
    
    clusters = list(cluster_map.values())
    
    # Clean up GPU resources
    del index
    del res
    
    clustering_elapsed = time.time() - clustering_start
    print(f"\n⚡ Clustering selesai dalam {clustering_elapsed:.2f} detik")
    
    # === SIMPAN HASIL ===
    idx_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for i in cluster:
            idx_to_cluster[i] = cluster_id
    
    print("\n💾 Saving results...")
    print(f"🎨 Drawing bounding boxes dengan {NUM_DRAWING_WORKERS} workers parallel...\n")
    
    # Split indices untuk parallel drawing (balanced distribution)
    all_indices = list(range(len(paths)))
    total_images = len(all_indices)
    base_chunk_size = total_images // NUM_DRAWING_WORKERS
    remainder = total_images % NUM_DRAWING_WORKERS
    
    # Buat proses parallel untuk drawing
    draw_start = time.time()
    draw_processes = []
    
    current_idx = 0
    for worker_id in range(NUM_DRAWING_WORKERS):
        # Workers dengan ID < remainder mendapat 1 image extra untuk balance
        chunk_size = base_chunk_size + (1 if worker_id < remainder else 0)
        worker_indices = all_indices[current_idx:current_idx + chunk_size]
        current_idx += chunk_size
        
        p = Process(target=draw_boxes_parallel, args=(worker_id, worker_indices, paths, face_boxes, idx_to_cluster))
        draw_processes.append(p)
        p.start()
    
    # Wait for all drawing workers
    for p in draw_processes:
        p.join()
    
    draw_elapsed = time.time() - draw_start
    print(f"\n⚡ Drawing selesai dalam {draw_elapsed:.2f} detik")
    
    # === SIMPAN KE REDIS ===
    if redis_client:
        print("\n📤 Saving clustering results to Redis...")
        
        clustering_result = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(paths),
            "total_clusters": len(clusters),
            "threshold": THRESHOLD,
            "processing_mode": "dual_gpu",
            "processing_time_seconds": elapsed,
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
        
        redis_key = "face_clustering:latest"
        redis_client.set(redis_key, json.dumps(clustering_result))
        
        history_key = f"face_clustering:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        redis_client.set(history_key, json.dumps(clustering_result))
        redis_client.expire(history_key, 7 * 24 * 60 * 60)
        
        print(f"   ✅ Saved to Redis key: {redis_key}")
        print(f"   ✅ Saved to Redis key: {history_key} (expires in 7 days)")
    
    total_time = elapsed + clustering_elapsed + draw_elapsed
    print(f"\n{'='*60}")
    print(f"✅ Clustering selesai!")
    print(f"   Total clusters: {len(clusters)} orang")
    print(f"   Extraction time: {elapsed:.2f}s ({len(paths)/elapsed:.2f} img/s)")
    print(f"   Clustering time: {clustering_elapsed:.2f}s (GPU-accelerated)")
    print(f"   Drawing time: {draw_elapsed:.2f}s ({len(paths)/draw_elapsed:.2f} img/s)")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Overall speed: {len(paths)/total_time:.2f} images/sec")
    print(f"📁 Hasil: {OUTPUT_DIR_LABELED}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
