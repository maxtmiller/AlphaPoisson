import torch
import torch.nn as nn
import numpy as np
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules
from chess import Board


def board_to_matrix(board: Board):
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1
    return matrix


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.skip_add.add_relu(out, x)

    def fuse_model(self):
        # conv2+bn2 can't absorb the final ReLU: the residual add sits between them (add_relu handles it).
        fuse_modules(self, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2']], inplace=True)


class ChessResNet(nn.Module):
    def __init__(self, num_res_blocks=4, num_moves=4672):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_relu = nn.ReLU()
        self.policy_fc = nn.Linear(32*8*8, num_moves)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        """
        x: [batch, 13, 8, 8] board input
        """
        out = self.stem(self.quant(x))
        out = self.res_blocks(out)
        policy = self.policy_relu(self.policy_bn(self.policy_conv(out)))
        policy = self.policy_fc(policy.flatten(1))
        return self.dequant(policy)

    def fuse_model(self):
        fuse_modules(self.stem, [['0', '1', '2']], inplace=True)
        for block in self.res_blocks:
            block.fuse_model()
        fuse_modules(self, [['policy_conv', 'policy_bn', 'policy_relu']], inplace=True)
    

class ChessResNetPa(nn.Module):
    def __init__(self, num_res_blocks=4, num_moves=4672):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_relu = nn.ReLU()
        self.policy_fc = nn.Linear(32*8*8, num_moves)

        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_relu = nn.ReLU()
        self.value_fc1 = nn.Linear(32*8*8, 128)
        self.value_fc1_relu = nn.ReLU()
        self.value_fc2 = nn.Linear(128, 1)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        """
        x: [batch, 13, 8, 8] board input
        """
        out = self.stem(self.quant(x))
        out = self.res_blocks(out)
        policy = self.policy_relu(self.policy_bn(self.policy_conv(out)))
        policy = self.policy_fc(policy.flatten(1))

        value = self.value_relu(self.value_bn(self.value_conv(out)))
        value = self.value_fc1_relu(self.value_fc1(value.flatten(1)))
        value = torch.tanh(self.dequant(self.value_fc2(value)))  # output in [-1, 1]

        return self.dequant(policy), value

    def fuse_model(self):
        fuse_modules(self.stem, [['0', '1', '2']], inplace=True)
        for block in self.res_blocks:
            block.fuse_model()
        fuse_modules(self, [['policy_conv', 'policy_bn', 'policy_relu'],
                            ['value_conv', 'value_bn', 'value_relu'],
                            ['value_fc1', 'value_fc1_relu']], inplace=True)

