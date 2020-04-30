import torch
import torch.nn
import torch.utils.data

from chess_recognizer.common import load_model
from chess_recognizer.training_data.boards import tensor_to_board
from chess_recognizer.training_data.torch_dataset import ChessImageDataset


def evaluate(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
):
    model.eval()
    model.cpu()
    for inputs, labels in data_loader:
        outputs = model(inputs)
        for i in range(labels.shape[0]):
            output_board = tensor_to_board(outputs[i])
            true_board = tensor_to_board(labels[i])
            print("true board\n", true_board)
            print("outputput board\n", output_board)
            print("******")


def main(model_name: str):
    model = load_model(model_name)
    dataloader = torch.utils.data.DataLoader(ChessImageDataset(count=2), batch_size=1,)
    evaluate(model, dataloader)


if __name__ == "__main__":
    main("resnet_untrained_2020.03.09.15.39.22.pt")
