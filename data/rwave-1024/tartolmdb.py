import tarfile
import lmdb
import cv2
from tqdm import tqdm
from io import BytesIO
import shutil
import numpy as np

def read_image_from_tar(tar, member):
    file = tar.extractfile(member)
    image_bytes = file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def check_disk_space(required_space):
    total, used, free = shutil.disk_usage("/")
    return free >= required_space

def convert_tar_to_lmdb(tar_gz_path, lmdb_path, initial_map_size=int(1e10), batch_size=1000, max_map_size=int(2e11)):
    def increase_map_size(env, factor=2):
        current_map_size = env.info()['map_size']
        new_map_size = current_map_size * factor
        if new_map_size > max_map_size:
            raise ValueError("Exceeded maximum map size limit.")
        env.set_mapsize(new_map_size)
        print(f"Map size increased to {new_map_size / 1e9} GB")

    env = lmdb.open(lmdb_path, map_size=initial_map_size)
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(('png', 'jpg', 'jpeg'))]
        num_batches = len(members) // batch_size + (1 if len(members) % batch_size != 0 else 0)
        
        for batch_idx in range(num_batches):
            while True:
                try:
                    with env.begin(write=True) as txn:
                        batch_members = members[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                        for idx, member in enumerate(tqdm(batch_members, desc=f"Processing batch {batch_idx + 1}/{num_batches}")):
                            image = read_image_from_tar(tar, member)
                            success, image_bytes = cv2.imencode('.png', image)
                            if not success:
                                raise ValueError("Failed to encode image as PNG.")
                            txn.put(f'image_{batch_idx * batch_size + idx}'.encode(), image_bytes.tobytes())
                    break
                except lmdb.MapFullError:
                    print("Map size full, increasing map size...")
                    increase_map_size(env)
    env.close()

# Example usage
tar_gz_path = 'rwave-1024.tar.gz'
lmdb_path = 'rwave-1024.lmdb'

# Check disk space before running the script
required_space = int(1e11)  # 100GB
if not check_disk_space(required_space):
    print("Not enough disk space. Please free up some space and try again.")
else:
    convert_tar_to_lmdb(tar_gz_path, lmdb_path)



