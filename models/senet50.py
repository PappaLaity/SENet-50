# import torch.nn as nn


# class SENet(nn.Module):
#     def __init__(
#         self, image_side_size, num_classes, in_channels=3
#     ):
#         super(SENet, self).__init__()

#         # transform images to 224*224
#         self.test = 121
#         conv_blocks = []
        
#         conv_blocks.append(
#             nn.Sequential(
#                 # input channel 256*256 --> 64*128*128 --> 64*64*64
#                 nn.Conv2d(in_channels, 64, 7, 2),
#                 nn.MaxPool2d(3, stride=2),
#             )
#         )
        
#         for _ in range(3):
#             conv_blocks.append(
#                 nn.Sequential(
#                     nn.Conv2d(64, 64, 1),
#                     nn.Conv2d(64, 64, 3),
#                     nn.Conv2d(64, 256, 1),
#                 )
#             )
#         for _ in range(4):
#             conv_blocks.append(
#                 nn.Sequential(
#                     nn.Conv2d(256, 128, 1),
#                     nn.Conv2d(128, 128, 3),
#                     nn.Conv2d(128, 512, 1),
#                 )
#             )
#         for _ in range(6):
#             conv_blocks.append(
#                 nn.Sequential(
#                     nn.Conv2d(512, 256, 1),
#                     nn.Conv2d(256, 256, 3),
#                     nn.Conv2d(256, 1024, 1),
#                 )
#             )
#         conv_blocks.append(
#                 nn.Sequential(
#                     nn.Conv2d(1024, 512,1,2),
#                     nn.Conv2d(512, 512, 3),
#                     nn.Conv2d(512, 2048, 1),
#                 )
#             )
#         conv_blocks.append(
#                 nn.Sequential(
#                     nn.Conv2d(2048, 512, 1),
#                     nn.Conv2d(512, 512, 3),
#                     nn.Conv2d(512, 2048, 1),
#                 )
#             )
#         conv_blocks.append(
#                 nn.Sequential(
#                     nn.Conv2d(2048, 512, 1),
#                     nn.Conv2d(512, 512, 3),
#                     nn.Conv2d(512, 2048, 1),
#                 )
#             )
#         self.conv_blocks = nn.ModuleList(conv_blocks)
#         self.linear = nn.Linear(self.test, num_classes)
#         # Loss CEntropyLoss
#         pass

#     def forward(self, x):
#         for block in self.conv_blocks:
#             x = self.conv_apply(block, x)
#             x += self.squeeze_excitation(x)
#         output = self.linear(x.view(x.size(0), -1))
#         return output

#     def conv_apply(self, block, x):
#         return block(x)

#     def squeeze_excitation(
#         self, x
#     ):  # X : C,H,W --(C,1,1 -> C/16,1,1 + ReLU -> C,1,1 + Sigmoid)--> C,H,W
#         # Squeeze step
#         # Average Pool for Channel

#         # Excitation Step (2 Fully Connected Layers)
#         # Dimensionnality Reduction (r = 16)
#         # Relu On Input Elements
#         # Dimensionality Increasing
#         # Sigmoid On the Results
#         # Return result with wich we element wise multiply by the Input..
#         pass
