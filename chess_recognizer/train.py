import torch.nn
import torch.utils.data
import torch.utils.tensorboard

from chess_recognizer.common import (
    get_curr_time_str,
    LOGS_DIR,
    BOARD_DIMENSIONS,
    save_model,
)
from chess_recognizer.models import Resnet
from chess_recognizer.training_data.torch_dataset import ChessImageDataset


def get_training_data(count: int = 10) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        ChessImageDataset(count=count, move_count=(30, 100)), batch_size=20,
    )


def train(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion=None,
    optimizer=None,
    cuda: bool = True,
):
    if criterion is None:
        criterion = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    summary_writer = torch.utils.tensorboard.SummaryWriter(
        LOGS_DIR / get_curr_time_str()
    )

    summary_writer.add_graph(
        model=model, input_to_model=torch.unsqueeze(data_loader.dataset[0][0], dim=0)
    )

    if cuda:
        model = model.cuda()

    for epoch in range(10):
        running_loss = 0.0
        total = 0
        correct = 0
        for i, (inputs, labels) in enumerate(data_loader, 1):
            total += labels.size(0)
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            truth = torch.argmax(
                labels.reshape(
                    labels.shape[0], BOARD_DIMENSIONS[0], BOARD_DIMENSIONS[1]
                ),
                dim=2,
            )
            predictions = torch.argmax(
                outputs.reshape(
                    outputs.shape[0], BOARD_DIMENSIONS[0], BOARD_DIMENSIONS[1]
                ),
                dim=2,
            )
            correct += torch.sum(truth == predictions).item()
            global_step = epoch * len(data_loader) + i
            summary_writer.add_scalar(
                tag="training loss",
                scalar_value=running_loss / total,
                global_step=global_step,
            )
            summary_writer.add_scalar(
                tag="training accuracy",
                scalar_value=correct / (total * 64),
                global_step=global_step,
            )
            if i % 100 == 0:
                print(
                    f"epoch={epoch}, i={i}, accuracy={correct / (total * BOARD_DIMENSIONS[0])}"
                )
    return model


def main():
    model = Resnet(pretrained=False)
    train_data = get_training_data(count=10000)
    model = train(model, train_data)
    save_model(model, "resnet_untrained")


if __name__ == "__main__":
    main()
