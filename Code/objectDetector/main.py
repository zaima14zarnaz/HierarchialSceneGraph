from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from Code.objectDetector.dataPrep import DataPrep
from Code.objectDetector.VisualGenomeDataset import VisualGenomeDataset
from Code.objectDetector.fasterrcnncustom import FasterRCNNCustom
import torch
import torch.nn as nn


from Code.objectDetector.train import Trainer


def custom_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])



    # Load annotations
    dataset_dir = "/Users/zaimazarnaz/PycharmProjects/HierarchicalSceneGraph/Dataset/visualGenome"
    images_dir = "/Users/zaimazarnaz/PycharmProjects/HierarchicalSceneGraph/Dataset/visualGenome/VG_100K"
    dataPrepper = DataPrep()
    annotations, label_map, num_classes = dataPrepper.parse_dataset(dataset_dir, images_dir, filename='objects.json')
    dataset = VisualGenomeDataset(images_dir, annotations, label_map, transform=transform)

    print("Dataset size: "+str(len(dataset)))

    # Split into training and validation lolo the lel
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)


    # Load pre-trained Faster R-CNN model
    model = FasterRCNNCustom(output_stride=8, BatchNorm=nn.BatchNorm2d, num_classes=num_classes, pretrained=True)
    model = model.create_model()

    # Replace the box predictor for the custom number of classes
    # in_features = model.roi_heads.box_predictor.cls_score.in_features  # Get the number of input features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_data_loader=train_loader,
        trainset=train_dataset,
        optimizer=optimizer,
        device=device,
        criterion=criterion
    )

    trainer.train(num_epochs=10)

