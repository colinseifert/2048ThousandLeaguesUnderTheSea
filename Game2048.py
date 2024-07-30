import random
import numpy as np
from Game2048NN import Game2048NN
from MoveGenerator import MoveGenerator
import torch

class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.model = Game2048NN()
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        # print("Initial board:")
        # self.print_board()

    def add_random_tile(self):
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def move_left(self):
        moved = False
        for i in range(4):
            tiles = [tile for tile in self.board[i] if tile != 0]
            new_row = []
            skip = False
            for j in range(len(tiles)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(tiles) and tiles[j] == tiles[j + 1]:
                    new_row.append(tiles[j] * 2)
                    self.score += tiles[j] * 2  # Update score
                    skip = True
                else:
                    new_row.append(tiles[j])
            new_row += [0] * (4 - len(new_row))
            if not np.array_equal(self.board[i], new_row):
                moved = True
            self.board[i] = new_row
        return moved

    def move_right(self):
        self.board = np.fliplr(self.board)
        moved = self.move_left()
        self.board = np.fliplr(self.board)
        return moved

    def move_up(self):
        self.board = np.rot90(self.board, 1)
        moved = self.move_left()
        self.board = np.rot90(self.board, -1)
        return moved

    def move_down(self):
        self.board = np.rot90(self.board, -1)
        moved = self.move_left()
        self.board = np.rot90(self.board, 1)
        return moved

    def make_move(self, direction):
        move_made = False
        if direction == 'UP':
            move_made = self.move_up()
        elif direction == 'DOWN':
            move_made = self.move_down()
        elif direction == 'LEFT':
            move_made = self.move_left()
        elif direction == 'RIGHT':
            move_made = self.move_right()
        # else:
        #     print(f"Invalid move: {direction}")

        if move_made:
            self.add_random_tile()
        return self.board

    def is_game_over(self):
        if any(0 in row for row in self.board):
            return False
        for i in range(4):
            for j in range(4):
                if (i < 3 and self.board[i][j] == self.board[i + 1][j]) or (
                    j < 3 and self.board[i][j] == self.board[i][j + 1]):
                    return False
        return True

    def get_score(self):
        return self.score

    def print_board(self):
        print(self.board)

    def run_game(self):
        
        while not self.is_game_over():
            try:
                move = MoveGenerator.generate_NN_moves(self.board, self.model)
                # print(f"Move: {move}")
            except Exception as e:
                print(f"Error during move generation: {e}")
                break
            self.make_move(move)
            # print("Board after move:")
            # self.print_board()
            # print(f"Score: {self.get_score()}")
        # print("Final board:")
        # self.print_board()
        # print(f"Final score: {self.get_score()}")
        return
