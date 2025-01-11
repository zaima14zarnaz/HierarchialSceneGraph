import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import json
import torch


class VisualGenomeDataset(Dataset):
    def __init__(self, image_dir, annotation_file, annotations_dir, transform=None):
        self.image_dir = image_dir
        self.annotations = json.load(open(f"{annotations_dir}/{annotation_file}", 'r'))
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = f"{self.image_dir}/{ann['image_id']}.jpg"
        image = Image.open(image_path).convert("RGB")

        boxes = ann['boxes']  # Bounding boxes
        labels = ann['labels']  # Object labels

        if self.transform:
            image = self.transform(image)

        target = {'boxes': torch.tensor(boxes, dtype=torch.float32),
                  'labels': torch.tensor(labels, dtype=torch.int64)}

        return image, target


