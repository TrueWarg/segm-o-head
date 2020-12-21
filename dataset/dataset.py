import os
import torch
from torch import nn
import xml.etree.ElementTree as ET
import cv2
from typing import Tuple

HEAD_DATASET = ""


class Dataset(nn.Module):
    def __init__(self, filepath: str, transformer=None):
        super().__init__()
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

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image, label
