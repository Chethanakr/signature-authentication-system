import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SiameseResNet18(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseResNet18, self).__init__()
        self.resnet = models.resnet18()
        # Single channel input for grayscale signatures
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward_one(self, x):
        out = self.resnet(x)
        return F.normalize(out, p=2, dim=1)

    def forward(self, img1, img2):
        return self.forward_one(img1), self.forward_one(img2)