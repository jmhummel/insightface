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
            torch.nn.Conv2d(256, 512, 2, padding=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(512, 1024, 5, 2, padding=2),
            torch.nn.LeakyReLU(0.1),
        )
        self.classifier = torch.nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.avg_pool2d(x, 9)
        x = torch.flatten(x)
        x = self.classifier(x)
        return x
