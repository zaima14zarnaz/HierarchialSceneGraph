import json
from Code.objectDetector2.VisualGenomeDataset import VisualGenomeDataset
from torchvision.transforms import transforms

# # Define data transforms
# transform = transforms.Compose([
#     transforms.Resize((600, 600)),
#     transforms.ToTensor(),
# ])
#
# # Load dataset
annotations_dir = "/Users/zaimazarnaz/PycharmProjects/HierarchicalSceneGraph/Dataset/visualGenome"
# images_dir = "/Users/zaimazarnaz/PycharmProjects/HierarchicalSceneGraph/Dataset/visualGenome/VG_100K"
annotations_file = "objects.json"
# dataset = VisualGenomeDataset(image_dir=images_dir, annotation_file=annotations_file, annotations_dir=annotations_dir, transform=transform)
annotations = json.load(open(f"{annotations_dir}/{annotations_file}", 'r'))
print(annotations[0])