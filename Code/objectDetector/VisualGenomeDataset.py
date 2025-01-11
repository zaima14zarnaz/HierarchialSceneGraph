# Dataloader class
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from Code.objectDetector.dataPrep import DataPrep



class VisualGenomeDataset(Dataset):
    def __init__(self, images_dir, annotations, label_map, transform=None):
        self.dataset_dir = images_dir
        self.annotations = annotations
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_id = annotation['image_id']
        label = annotation['label']
        bbox = annotation['bbox']


        # Ensure label and bbox are lists
        if isinstance(label, int):  # Single label
            label = [label]
        if isinstance(bbox[0], (int, float)):  # Single bbox
            bbox = [bbox]

        # Debugging outputs
        print(f"Image ID: {img_id}")
        print(f"Label: {label}, BBox: {bbox}")

        # Load the image
        img_path = os.path.join(self.dataset_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert to tensors
        target = {'labels': torch.tensor(label, dtype=torch.int64),
                  'boxes': torch.tensor(bbox, dtype=torch.float32)}

        return image, target

    def label_to_idx(self, label):
        return self.label_map[label]


# if __name__ == "__main__":
#     def custom_collate_fn(batch):
#         """
#         Custom collate function for handling batches of images and targets
#         in object detection datasets.
#         """
#         images = [item[0] for item in batch]  # Extract images
#         targets = [item[1] for item in batch]  # Extract targets
#         return images, targets
#
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize to model input size
#         transforms.ToTensor(),          # Convert to tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
#     ])
#
#
#
#     # Load annotations
#     dataset_dir = "/Users/zaimazarnaz/PycharmProjects/HierarchicalSceneGraph/Dataset/visualGenome"
#     images_dir = "/Users/zaimazarnaz/PycharmProjects/HierarchicalSceneGraph/Dataset/visualGenome/VG_100K"
#     dataPrepper = DataPrep()
#     annotations, label_map, num_classes = dataPrepper.parse_dataset(dataset_dir, images_dir, filename='objects.json')
#     dataset = VisualGenomeDataset(images_dir, annotations, label_map, transform=transform)
#
#     print("Dataset size: "+str(len(dataset)))
#
#     # Split into training and validation
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
#     # Create DataLoaders
#     # Initialize the DataLoader
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=8,
#         shuffle=True,
#         num_workers=1,
#         collate_fn=custom_collate_fn
#     )
#     val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)
#
#     for idx, (images, targets) in enumerate(train_loader):
#         print(f"Batch Index: {idx}")
#         print(f"Number of Images in Batch: {len(images)}")
#         print(f"Targets: {targets}")
#
#         # Example of accessing individual target fields
#         for target in targets:
#             print(f"Labels: {target['labels']}")
#             print(f"Bounding Boxes: {target['bbox']}")

