#!/usr/bin/env python3
"""
Script untuk membaca hasil clustering dari Redis
"""
import redis
import json
from datetime import datetime

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Baca hasil terbaru
print("📊 HASIL CLUSTERING TERBARU\n" + "="*50)

data_str = r.get('face_clustering:latest')
if not data_str:
    print("❌ Tidak ada data clustering di Redis")
    exit(1)

result = json.loads(data_str)

print(f"⏰ Timestamp: {result['timestamp']}")
print(f"📸 Total Images: {result['total_images']}")
print(f"👥 Total Clusters: {result['total_clusters']}")
print(f"🎯 Threshold: {result['threshold']}")
print("\n" + "="*50 + "\n")

# Tampilkan detail per cluster
print("📋 DETAIL PER CLUSTER:\n")
for cluster in result['clusters']:
    print(f"  🏷️  {cluster['person_label']}: {cluster['image_count']} foto")

print("\n" + "="*50 + "\n")

# Contoh: Ambil cluster dengan foto terbanyak
max_cluster = max(result['clusters'], key=lambda x: x['image_count'])
print(f"👑 Cluster dengan foto terbanyak: {max_cluster['person_label']}")
print(f"   Total: {max_cluster['image_count']} foto")
print(f"   Files:")
for img in max_cluster['images'][:3]:  # Tampilkan 3 pertama
    print(f"   - {img['filename']}")
if max_cluster['image_count'] > 3:
    print(f"   ... dan {max_cluster['image_count'] - 3} foto lainnya")

print("\n" + "="*50 + "\n")

# Tampilkan history keys
print("📜 HISTORY KEYS:")
keys = r.keys('face_clustering:*')
for key in sorted(keys):
    if key != 'face_clustering:latest':
        ttl = r.ttl(key)
        print(f"   - {key} (expires in {ttl}s / {ttl//3600}h)")
