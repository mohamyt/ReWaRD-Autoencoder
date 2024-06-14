import lmdb
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class Dataset_(Dataset):
    def __init__(self, lmdb_file, transform=None):
        self.transform = transform
        self.lmdb_file = lmdb_file

        self.keys = []
        with lmdb.open(self.lmdb_file, readonly=True, lock=False) as env:
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    self.keys.append(key)

        # Class-level cache for the LMDB environment
        self.env = None

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_file, readonly=True, lock=False)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):    
        self._init_env()
        key = self.keys[idx]
        with self.env.begin() as txn:
            value = txn.get(key)
            # Decode the image
            image = np.frombuffer(value, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image
