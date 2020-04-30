import torch
import torch.nn
import torchvision.models
import torchvision.models.resnet

from chess_recognizer.common import BOARD_DIMENSIONS


class Resnet(torch.nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        if pretrained:
            for param in self.resnet.parameters():
                param.requires_grad = True

        self.resnet.fc = torch.nn.Linear(
            in_features=512 * torchvision.models.resnet.BasicBlock.expansion,
            out_features=BOARD_DIMENSIONS[0] * BOARD_DIMENSIONS[1],
        )

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet.forward(x)
