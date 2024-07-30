import numpy as np
import torch

class MoveGenerator:
    UP = 'UP'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'

    def __init__(self):
        pass
        
    @staticmethod
    def generate_NN_moves(gameGrid, model) -> str:
        # Normalize the game grid using a logarithmic transformation
        normalized_grid = np.where(gameGrid > 0, np.log2(gameGrid), 0)
        # Min-Max normalization to scale values between 0 and 1
        min_val = np.min(normalized_grid)
        max_val = np.max(normalized_grid)
        if max_val > min_val:  # To avoid division by zero
            normalized_grid = (normalized_grid - min_val) / (max_val - min_val)
        # Flatten the normalized game grid and convert to a PyTorch tensor
        input_tensor = torch.tensor(normalized_grid.flatten(), dtype=torch.float32)
        output_tensor = model(input_tensor)
        
        # Print the model output for debugging
        # print(f"Model output: {output_tensor}")

        move_indices = torch.argsort(output_tensor, descending=True).tolist()

        for move_index in move_indices:
            move = MoveGenerator.index_to_move(move_index)
            if MoveGenerator.is_valid_move(gameGrid, move):
                # print(f"NN returned {move}")
                return move

        raise Exception("No valid moves found")
        
    @staticmethod
    def index_to_move(index: int) -> str:
        if index == 0:
            return MoveGenerator.UP
        elif index == 1:
            return MoveGenerator.DOWN
        elif index == 2:
            return MoveGenerator.LEFT
        elif index == 3:
            return MoveGenerator.RIGHT
        else:
            raise ValueError(f"Invalid move index: {index}")

    @staticmethod
    def is_valid_move(gameGrid, move: str) -> bool:
        temp_grid = gameGrid.copy()
        if move == MoveGenerator.UP:
            temp_grid = MoveGenerator.slide_up(temp_grid)
        elif move == MoveGenerator.DOWN:
            temp_grid = MoveGenerator.slide_down(temp_grid)
        elif move == MoveGenerator.LEFT:
            temp_grid = MoveGenerator.slide_left(temp_grid)
        elif move == MoveGenerator.RIGHT:
            temp_grid = MoveGenerator.slide_right(temp_grid)

        return not np.array_equal(temp_grid, gameGrid)

    @staticmethod
    def slide_up(grid):
        for col in range(4):
            new_col = [num for num in grid[:, col] if num != 0]
            new_col += [0] * (4 - len(new_col))
            for i in range(3):
                if new_col[i] == new_col[i + 1] and new_col[i] != 0:
                    new_col[i] *= 2
                    new_col[i + 1] = 0
            new_col = [num for num in new_col if num != 0]
            new_col += [0] * (4 - len(new_col))
            grid[:, col] = new_col
        return grid

    @staticmethod
    def slide_down(grid):
        for col in range(4):
            new_col = [num for num in grid[:, col] if num != 0]
            new_col = [0] * (4 - len(new_col)) + new_col
            for i in range(3, 0, -1):
                if new_col[i] == new_col[i - 1] and new_col[i] != 0:
                    new_col[i] *= 2
                    new_col[i - 1] = 0
            new_col = [num for num in new_col if num != 0]
            new_col = [0] * (4 - len(new_col)) + new_col
            grid[:, col] = new_col
        return grid

    @staticmethod
    def slide_left(grid):
        for row in range(4):
            new_row = [num for num in grid[row, :] if num != 0]
            new_row += [0] * (4 - len(new_row))
            for i in range(3):
                if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                    new_row[i] *= 2
                    new_row[i + 1] = 0
            new_row = [num for num in new_row if num != 0]
            new_row += [0] * (4 - len(new_row))
            grid[row, :] = new_row
        return grid

    @staticmethod
    def slide_right(grid):
        for row in range(4):
            new_row = [num for num in grid[row, :] if num != 0]
            new_row = [0] * (4 - len(new_row)) + new_row
            for i in range(3, 0, -1):
                if new_row[i] == new_row[i - 1] and new_row[i] != 0:
                    new_row[i] *= 2
                    new_row[i - 1] = 0
            new_row = [num for num in new_row if num != 0]
            new_row = [0] * (4 - len(new_row)) + new_row
            grid[row, :] = new_row
        return grid
