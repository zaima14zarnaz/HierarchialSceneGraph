# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.backbone_utils import BackboneWithFPN
# from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
# from torchvision.models.detection.rpn import AnchorGenerator
# from collections import OrderedDict
# import torch
# import torch.nn as nn
# from torchvision.models import resnet152
#
# class ResnetBackbone:
#     # Create ResNet-152 backbone
#     def get_resnet152_backbone(self):
#         resnet152_model = resnet152(weights="DEFAULT")
#         return resnet152_model
#
#     # Create Faster R-CNN model
#     def create_faster_rcnn_resnet152(self, num_classes):
#         backbone = self.get_resnet152_backbone()
#         backbone_with_fpn = ResNetFPNBackbone(backbone, out_channels=256)
#
#         # Define anchor generator
#         anchor_generator = AnchorGenerator(
#             sizes=((32,), (64,), (128,), (256,)),
#             aspect_ratios=((0.5, 1.0, 2.0),) * 4
#         )
#
#         # Create Faster R-CNN with the custom backbone
#         model = FasterRCNN(
#             backbone=ResNetFPNBackbone(resnet152(weights="DEFAULT"), 256),
#             num_classes=num_classes,  # Number of classes including background
#             rpn_anchor_generator=anchor_generator
#         )
#         return model
#
# class ResNetFPNBackbone(nn.Module):
#     def __init__(self, backbone, out_channels):
#         super().__init__()
#         # Include ResNet's initial layers
#         self.conv1 = backbone.conv1
#         self.bn1 = backbone.bn1
#         self.relu = backbone.relu
#         self.maxpool = backbone.maxpool
#
#         # Extract specific layers
#         self.layer1 = backbone.layer1  # Layer2 of ResNet
#         self.layer2 = backbone.layer2  # Layer3 of ResNet
#         self.layer3 = backbone.layer3  # Layer4 of ResNet
#         self.layer4 = backbone.layer4  # Final ResNet layer
#
#         # Create FPN
#         self.fpn = FeaturePyramidNetwork(
#             in_channels_list=[256, 512, 1024, 2048],  # Channels from ResNet layers
#             out_channels=out_channels
#         )
#
#         self.out_channels = out_channels  # Required by FasterRCNN
#
#     def forward(self, x):
#         # Pass input through the initial ResNet layers
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         # Pass input through ResNet layers
#         c2 = self.layer1(x)  # Output of ResNet layer2
#         c3 = self.layer2(c2)  # Output of ResNet layer3
#         c4 = self.layer3(c3)  # Output of ResNet layer4
#         c5 = self.layer4(c4)  # Final feature map
#
#         # Create a dictionary for FPN
#         features_dict = {
#             "feat2": c2,
#             "feat3": c3,
#             "feat4": c4,
#             "feat5": c5
#         }
#
#         # Pass through the FPN
#         fpn_features = self.fpn(features_dict)
#
#         return fpn_features
#
# # Create Faster R-CNN
# resnet = ResnetBackbone()
# model = resnet.create_faster_rcnn_resnet152(num_classes=151)
#
# # Dummy input
# images = [torch.rand(3, 224, 224) for _ in range(4)]
# targets = [{
#     "boxes": torch.tensor([[50, 50, 100, 100]], dtype=torch.float32),
#     "labels": torch.tensor([1], dtype=torch.int64)
# } for _ in range(4)]
#
# # Forward pass
# model.train()
# loss_dict = model(images, targets)
# print("Loss Dict:", loss_dict)
#
