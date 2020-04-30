from typing import Iterable, Tuple, Optional, Any

import torch.utils.data
import torchvision

from chess_recognizer.training_data.boards import (
    yield_valid_boards,
    board_to_tensor,
    board_to_pil,
)


class ChessImageDataset(torch.utils.data.dataset.IterableDataset):
    def __init__(
        self,
        count: int,
        move_count: Tuple[int, int] = (0, 100),
        transform: Optional[Any] = None,
        label_dtype=torch.float,
    ) -> None:
        super().__init__()
        self.count = count
        self.move_count = move_count
        if transform is None:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ColorJitter(),
                    torchvision.transforms.RandomGrayscale(),
                    torchvision.transforms.Resize(size=(224, 224)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        self.transform = transform
        self.label_dtype = label_dtype

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        for board in yield_valid_boards(count=self.count, move_count=self.move_count,):
            x = board_to_pil(board)
            y = board_to_tensor(board, dtype=self.label_dtype)
            yield self.transform(x), y

    def __len__(self):
        return self.count

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        board = next(iter(yield_valid_boards(count=1)))
        x = board_to_pil(board)
        y = board_to_tensor(board, dtype=self.label_dtype)
        return self.transform(x), y


def main():
    for image, tensor in ChessImageDataset(count=1, move_count=(20, 100)):
        torchvision.transforms.ToPILImage()(image).show()


if __name__ == "__main__":
    main()
