import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import itertools
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Dataset_(Dataset):
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def load_image(self, index):
        image_path = self.image_files[index]
        return cv2.imread(image_path)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        img = cv2.imread(image_path)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img

    def get_image(self, index):
        return self.load_image(self.image_files[index])
