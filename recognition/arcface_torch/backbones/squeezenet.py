from torch import nn
from torch.nn import init
from torchvision.models.squeezenet import squeezenet1_1, Fire


def get_squeezenet_112(num_features=512):
    model = squeezenet1_1(num_classes=num_features)
    model.features = nn.Sequential(
        nn.Conv2d(3, 128, kernel_size=3, stride=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        Fire(128, 32, 128, 128),
        Fire(256, 32, 128, 128),
        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        Fire(256, 48, 192, 192),
        Fire(384, 48, 192, 192),
        Fire(384, 64, 256, 256),
        Fire(512, 64, 256, 256),
    )
    for m in model.features:
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
    return model