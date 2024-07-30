

import numpy
import random
import torch
from selenium.webdriver.common.keys import Keys

UP = Keys.ARROW_UP
DOWN = Keys.ARROW_DOWN
LEFT = Keys.ARROW_LEFT
RIGHT = Keys.ARROW_RIGHT

class MoveGenerator:
    def __init__(self):
        return
        
    @staticmethod
    def generate_NN_moves(gameGrid, model) -> str:
        # Normalize the game grid using a logarithmic transformation
        normalized_grid = numpy.where(gameGrid > 0, numpy.log2(gameGrid), 0)
        # Min-Max normalization to scale values between 0 and 1
        min_val = numpy.min(normalized_grid)
        max_val = numpy.max(normalized_grid)
        if max_val > min_val:  # To avoid division by zero
            normalized_grid = (normalized_grid - min_val) / (max_val - min_val)
        # Flatten the normalized game grid and convert to a PyTorch tensor
        input_tensor = torch.tensor(normalized_grid.flatten(), dtype=torch.float32)
        output_tensor = model(input_tensor)
        # print(output_tensor)
        move_indices = torch.argsort(output_tensor, descending=True).tolist()

        for move_index in move_indices:
            move = MoveGenerator.index_to_move(move_index)
            if MoveGenerator.is_valid_move(gameGrid, move):
                # print(f"NN returned {MoveGenerator.move_to_readable_string(move)}")
                return move
            # print("NN returned invalid moves, falling back to next best move")

        # fail gracefully if no valid moves found
        raise Exception("No valid moves found")
        
    @staticmethod
    def move_to_readable_string(move: str) -> str:
        if move == UP:
            return "UP"
        elif move == DOWN:
            return "DOWN"
        elif move == LEFT:
            return "LEFT"
        else:
            return "RIGHT"
    
    @staticmethod
    def index_to_move(index: int) -> str:
        if index == 0:
            return UP
        elif index == 1:
            return DOWN
        elif index == 2:
            return LEFT
        else:
            return RIGHT

    @staticmethod
    def is_valid_move(gameGrid, move: str) -> bool:
        # Check if the move is valid by simulating the move and checking for changes in the grid
        temp_grid = gameGrid.copy()
        if move == UP:
            temp_grid = MoveGenerator.slide_up(temp_grid)
        elif move == DOWN:
            temp_grid = MoveGenerator.slide_down(temp_grid)
        elif move == LEFT:
            temp_grid = MoveGenerator.slide_left(temp_grid)
        elif move == RIGHT:
            temp_grid = MoveGenerator.slide_right(temp_grid)

        return not numpy.array_equal(temp_grid, gameGrid)

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

    @staticmethod
    def generate_random_move(gameGrid) -> str:
        valid_moves = [UP, DOWN, LEFT, RIGHT]
        random.shuffle(valid_moves)
        for move in valid_moves:
            if MoveGenerator.is_valid_move(gameGrid, move):
                return move
        return UP  # This should never happen unless the game is over
    
    