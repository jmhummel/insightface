import torch.nn


class SimpleEncoder(torch.nn.Module):
    def __init__(self, num_classes=512):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 5, 2, padding=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(64, 128, 5, 2, padding=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(128, 256, 5, 2, padding=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(256, 512, 5, 2, padding=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(512, 1024, 5, 2, padding=2),
            torch.nn.LeakyReLU(0.1),
        )
        self.flatten = torch.nn.Flatten()
        self.classifier = torch.nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.avg_pool2d(x, 4)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class SimpleEncoder2(torch.nn.Module):
    def __init__(self, num_classes=512):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 512, 3),
            torch.nn.LeakyReLU(0.2),
        )
        self.flatten = torch.nn.Flatten()
        self.classifier = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x