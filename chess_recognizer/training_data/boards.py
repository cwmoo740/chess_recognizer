import io
import logging
import random
from pathlib import Path
from typing import Iterable

import PIL.Image
import cairosvg
import chess
import chess.svg
import torch

from chess_recognizer.common import BOARD_DIMENSIONS

DATA_FOLDER = Path(__file__).parent / ".." / ".." / "data"

LOGGER = logging.getLogger(__name__)


def yield_valid_boards(count: int,) -> Iterable[chess.Board]:
    """
    places a single white knight on an empty board at a random position
    Args:
        count: the number of boards to yield

    Returns:

    """
    for i in range(count):
        board = chess.Board(fen="")
        board.set_piece_at(
            square=random.choice(chess.SQUARES),
            piece=chess.Piece(piece_type=chess.KNIGHT, color=chess.WHITE),
        )
        yield board


def board_to_pil(board: chess.Board) -> PIL.Image.Image:
    svg = chess.svg.board(board, coordinates=False)
    output = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
    return PIL.Image.open(io.BytesIO(output))


def board_to_tensor(board: chess.Board, dtype=torch.float) -> torch.Tensor:
    output = torch.zeros(size=BOARD_DIMENSIONS, dtype=dtype)
    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        offset = chess.KING if piece.color else 0
        output[sq][piece.piece_type + offset] = True
    for i in range(BOARD_DIMENSIONS[0]):
        if i not in piece_map:
            output[i][0] = True
    return output.flatten()


def tensor_to_board(tensor: torch.Tensor) -> chess.Board:
    """

    :param tensor: a tensor representing a board state, 64*6
    :return:
    """
    assert tensor.shape == (BOARD_DIMENSIONS[0] * BOARD_DIMENSIONS[1],)
    tensor = tensor.reshape(BOARD_DIMENSIONS)
    board = chess.Board(fen=None)
    for i, j in enumerate(torch.argmax(tensor, dim=1)):
        if j.item() == 0:
            continue
        color = chess.WHITE if j > 6 else chess.BLACK
        board.set_piece_at(
            square=i, piece=chess.Piece(((j - 1) % 6) + 1, color),
        )
    return board


def main():
    for board in yield_valid_boards(10):
        board.reset()
        tensor = board_to_tensor(board)
        new_board = tensor_to_board(tensor)
        assert board.board_fen() == new_board.board_fen()


if __name__ == "__main__":
    main()
