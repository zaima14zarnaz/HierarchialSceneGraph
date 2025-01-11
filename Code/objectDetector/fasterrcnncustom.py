import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from openai import images

from Code.objectDetector.batchnorm import SynchronizedBatchNorm2d

import os
from torchvision import transforms
from PIL import Image

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )

class FasterRCNNCustom():
    def __init__(self, output_stride, BatchNorm, num_classes, pretrained=True):
        self.output_stride = output_stride
        self.num_classes = num_classes
        self.BatchNorm = BatchNorm
        self.pretrained = pretrained


    def create_model(self):
        # Create ResNet backbone
        backbone = ResNet101(self.output_stride, self.BatchNorm, pretrained=self.pretrained)
        backbone.out_channels = 2048  # Last layer's output channels

        # Define RPN Anchor Generator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),  # Anchor sizes
            aspect_ratios=((0.5, 1.0, 2.0),) * 5  # Anchor aspect ratios
        )

        # Define ROI Pooling
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],  # Name of the feature map(s) to use
            output_size=7,  # Size of the pooled region
            sampling_ratio=2  # Sampling ratio for ROI align
        )

        # Create Faster R-CNN
        model = FasterRCNN(
            backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )
        return model

# if __name__ == "__main__":
#     import torch
#     model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize to model input size
#         transforms.ToTensor(),  # Convert to tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
#     ])
#     images_dir = "/Users/zaimazarnaz/PycharmProjects/HierarchicalSceneGraph/Dataset/visualGenome/VG_100K"
#
#     img_path = os.path.join(images_dir, f"{2}.jpg")
#     image = Image.open(img_path).convert("RGB")
#     print(image.size)
#     image = transform(image)
#     image = image.unsqueeze(0)
#     print(image.size)
#
#     targets = [{
#         "boxes": torch.tensor([[50, 50, 100, 100]], dtype=torch.float32),
#         "labels": torch.tensor([1], dtype=torch.int64)
#     } for _ in range(4)]
#     output, low_level_feat = model(image)
#     print(output.size())
#     print(low_level_feat.size())
#
#     losses = nn.CrossEntropyLoss(output, targets)
#     print(losses.state_dict())

# if __name__ == "__main__":
#     # Create the Faster R-CNN model
#     num_classes = 21  # Example: 20 object classes + background
#     model = FasterRCNNCustom(output_stride=16, BatchNorm=nn.BatchNorm2d, num_classes=num_classes, pretrained=True)
#     model = model.create_model()
#
#     # Dummy input
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize to model input size
#         transforms.ToTensor(),  # Convert to tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
#     ])
#     images_dir = "/Users/zaimazarnaz/PycharmProjects/HierarchicalSceneGraph/Dataset/visualGenome/VG_100K"
#     img_path = os.path.join(images_dir, f"{2}.jpg")
#     image = Image.open(img_path).convert("RGB")
#     print(image.size)
#     image = transform(image)
#     image = image.unsqueeze(0)
#     print(image.size)
#
#     targets = [{
#         "boxes": torch.tensor([[50, 50, 100, 100]], dtype=torch.float32),
#         "labels": torch.tensor([1], dtype=torch.int64)
#     } for _ in range(4)]
#
#     # Set model to train mode
#     model.train()
#     loss_dict = model(image, targets)  # Forward pass
#
#
#     # Print losses
#     print(loss_dict)
#
#     # Set model to eval mode
#     model.eval()
#     with torch.no_grad():
#         predictions = model(image)  # Predictions for multiple objects
#         print(predictions)