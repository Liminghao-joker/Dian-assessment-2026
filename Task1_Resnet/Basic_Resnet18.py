import torch
from torch import nn

# BasicBlock for ResNet-18
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # the first Residual Block
        # probably dimension not match
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # the second Residual Block
        # dimension keeps same
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # the first convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # the second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # to solve conv3-5 dimension mismatch
        if self.downsample is not None:
            identity = self.downsample(x)

        # residual connection
        out += identity
        out = self.relu(out) # activation function
        return out

# ResNet-18
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000, cifar_stem = False):
        super(ResNet18, self).__init__()
        """
        after conv1 layer
        input:(3, 224, 224)
        output:(64, 56, 56)
        """
        if cifar_stem:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = None
        else:
            # Cov1
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # max pooling
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        """
        after residual layers
        input:(64, 56, 56)
        output:(512, 7, 7)
        """
        # residual layers
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # (512, 7, 7) -> (512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes) # 512 -> num_classes

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        #? 为什么此处要加入 stride!=1 的条件
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        # only focus on the first block
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        # the rest blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        # conv1 layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        # residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # fully connected layer
        x = self.fc(x)

        return x

def resnet18_cifar(num_classes=10):
    return ResNet18(num_classes=num_classes, cifar_stem = True)

if __name__ == "__main__":
    model = ResNet18(num_classes=1000)
    print(model)
