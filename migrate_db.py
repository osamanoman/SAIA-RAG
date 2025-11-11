#!/usr/bin/env python3
"""
Migrate Qdrant database from local to server using snapshot upload API
Best practice method from official Qdrant documentation
"""
import requests
import os
import sys
from qdrant_client import QdrantClient

# Configuration
LOCAL_QDRANT = "http://qdrant:6333"  # Docker internal network
SERVER_QDRANT = "http://134.209.10.163:6333"
COLLECTION_NAME = "docs_t_customerA"
SNAPSHOT_DIR = "/tmp/snapshots"  # Use /tmp in container

def create_snapshot(client, collection_name):
    """Create a snapshot of the collection"""
    print(f"Creating snapshot for collection '{collection_name}'...")
    snapshot_info = client.create_snapshot(collection_name=collection_name)
    print(f"✅ Snapshot created: {snapshot_info.name}")
    return snapshot_info.name

def download_snapshot(qdrant_url, collection_name, snapshot_name, output_path):
    """Download snapshot file"""
    print(f"Downloading snapshot '{snapshot_name}'...")
    snapshot_url = f"{qdrant_url}/collections/{collection_name}/snapshots/{snapshot_name}"

    response = requests.get(snapshot_url, stream=True)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Downloaded: {output_path} ({file_size_mb:.2f} MB)")
    return output_path

def upload_snapshot(qdrant_url, collection_name, snapshot_path):
    """Upload snapshot to target Qdrant instance"""
    print(f"Uploading snapshot to server...")
    upload_url = f"{qdrant_url}/collections/{collection_name}/snapshots/upload?priority=snapshot"

    with open(snapshot_path, 'rb') as f:
        files = {'snapshot': (os.path.basename(snapshot_path), f)}
        response = requests.post(upload_url, files=files)
        response.raise_for_status()

    print(f"✅ Snapshot uploaded successfully")
    return response.json()

def verify_migration(qdrant_url, collection_name):
    """Verify the collection after migration"""
    print(f"Verifying collection '{collection_name}'...")
    client = QdrantClient(url=qdrant_url)

    collection_info = client.get_collection(collection_name=collection_name)
    points_count = collection_info.points_count

    print(f"✅ Collection verified: {points_count} documents")
    return points_count

def main():
    print("=== Qdrant Database Migration (Best Practice Method) ===\n")

    # Create snapshots directory
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    try:
        # Step 1: Connect to local Qdrant
        print("1. Connecting to local Qdrant...")
        local_client = QdrantClient(url=LOCAL_QDRANT)
        local_info = local_client.get_collection(collection_name=COLLECTION_NAME)
        print(f"   Local: {local_info.points_count} documents\n")

        if local_info.points_count == 0:
            print("❌ Error: Local database is empty!")
            sys.exit(1)

        # Step 2: Create snapshot on local
        print("2. Creating snapshot on local Qdrant...")
        snapshot_name = create_snapshot(local_client, COLLECTION_NAME)
        print()

        # Step 3: Download snapshot
        print("3. Downloading snapshot...")
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_name)
        download_snapshot(LOCAL_QDRANT, COLLECTION_NAME, snapshot_name, snapshot_path)
        print()

        # Step 4: Upload to server
        print("4. Uploading snapshot to server...")
        upload_snapshot(SERVER_QDRANT, COLLECTION_NAME, snapshot_path)
        print()

        # Step 5: Verify migration
        print("5. Verifying migration...")
        server_points = verify_migration(SERVER_QDRANT, COLLECTION_NAME)
        print()

        # Final summary
        print("=" * 50)
        print("✅ MIGRATION COMPLETE!")
        print(f"   Source: {local_info.points_count} documents")
        print(f"   Target: {server_points} documents")
        print(f"   Status: {'SUCCESS' if server_points == local_info.points_count else 'MISMATCH'}")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Migration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

