from torch import nn
from torchvision.models import resnet18


def get_resnet(pretrained: bool=True, num_classes: int=10):
    model = resnet18(pretrained=pretrained)
    conv1 = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=False)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    return model
