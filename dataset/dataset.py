import os
import torch
import xml.etree.ElementTree as ET
import cv2
from typing import Tuple
from torch.utils.data import Dataset
import albumentations as A

HEAD_DATASET = ""


class Transformer():
    def __init__(self):
        self.image_transformations = self._create_image_transformations()
        self.label_transformations = self._create_label_transformations()

    def transform(self, image: np.ndarray, label: np.ndarray):
        transformed_image_result = self.image_transformations(image=image)
        transformed_label_result = self.label_transformations(image=label)
        return transformed_image_result['image'], transformed_label_result['image']

    def _create_image_transformations(self):
        return A.Compose(
            [
                A.Resize(width=256, height=256),
                A.ToFloat(max_value=255, always_apply=True)
            ]
        )

    def _create_label_transformations(self):
        return A.Resize(width=256, height=256)


class Dataset(Dataset):
    def __init__(self, filepath: str, transformer=None):
        self.transformer = transformer
        tree = ET.parse(filepath)
        root = tree.getroot()

        self.image_paths = []
        self.label_paths = []

        for child in root:
            if child.tag == 'srcimg':
                self.image_paths.append(os.path.join(HEAD_DATASET, child.attrib['name'].replace('\\', '/')))

            if child.tag == 'labelimg':
                self.label_paths.append(os.path.join(HEAD_DATASET, child.attrib['name'].replace('\\', '/')))

    def __getitem__(self, index: int) -> Tuple:
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(self.label_paths[index])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        if self.transformer != None:
            image, label = self.transformer.transform(image, label)

        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.from_numpy(label).permute(2, 0, 1)

        return image, label

    def __len__(self):
        return len(self.image_paths)
