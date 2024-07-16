import random
import re
import time

import numpy
import torch
import torch.nn
import torch.nn.functional as F
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

UP = Keys.ARROW_UP
DOWN = Keys.ARROW_DOWN
LEFT = Keys.ARROW_LEFT
RIGHT = Keys.ARROW_RIGHT


class utility:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.gameGrid = numpy.zeros((4, 4))
        self.model = Game2048NN()
        return

    def runGame(self, numMoves):
        self.driver.get("https://play2048.co/")
        selector = "//*[contains(@class, 'tile tile-')]"
        waitCond = WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, selector)))  # stored as a variable for debugging?
        stopCondition = "body > div.container.upper > div.game-container > div.game-message.game-over"
        while (len(self.driver.find_elements(By.CSS_SELECTOR, stopCondition)) == 0 and numMoves != 0):
            waitCond = WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, selector)))
            self.getGameState()
            self.printGameGrid()
            move = self.generateNNMove()
            self.sendMove(move)
            numMoves -= 1
        finalState = self.getGameState()
        self.printGameGrid()
        self.driver.quit()
        return finalState

    def getGameState(self):
        row = 0
        col = 0
        selector = "//*[contains(@class, 'tile tile-')]"

        time.sleep(0.2)  # wait for page to catch up after each move
        elements = self.driver.find_elements(By.XPATH, selector)
        # print(elements)
        self.gameGrid = numpy.zeros((4, 4))
        for element in elements:
            # regex the string to get coordinates
            # convert text to int
            # update matrix with int at coordinates
            coordinates = re.search(r"tile-position-(\d+)-(\d+)", element.get_attribute('class'))
            row = int(coordinates.group(2)) - 1  # decrement to have 0 indexing
            col = int(coordinates.group(1)) - 1  # "  "   "   "   "   "   "
            self.gameGrid[row][col] = int(element.text)

    def generateRandomMove(self) -> str:
        num = random.randint(1, 4)
        # print("randomMove:",num)
        if (num == 1):
            return UP
        if (num == 2):
            return DOWN
        if (num == 3):
            return LEFT
        return RIGHT

    def generateNNMove(self) -> str:
        # Flatten the game grid and convert to a PyTorch tensor
        input_tensor = torch.tensor(self.gameGrid.flatten(), dtype=torch.float32)
        output_tensor = self.model(input_tensor)
        move_index = torch.argmax(output_tensor).item()
        if move_index == 0:
            print("NN returned UP")
            return UP
        elif move_index == 1:
            print("NN returned DOWN")
            return DOWN
        elif move_index == 2:
            print("NN returned LEFT")
            return LEFT
        else:
            print("NN returned RIGHT")
            return RIGHT

    def sendMove(self, move):
        body = self.driver.find_element(By.TAG_NAME, 'body')
        body.send_keys(move)
        return

    def printGameGrid(self):
        print(self.gameGrid)
        return


class Game2048NN(torch.nn.Module):
    def __init__(self):
        super(Game2048NN, self).__init__()
        # Define the input layer (16 inputs) and the first hidden layer
        self.fc1 = torch.nn.Linear(16, 128)  # First hidden layer with 128 neurons
        self.fc2 = torch.nn.Linear(128, 64)  # Second hidden layer with 64 neurons
        self.fc3 = torch.nn.Linear(64, 32)  # Third hidden layer with 32 neurons
        self.fc4 = torch.nn.Linear(32, 4)  # Output layer with 4 neurons (one for each move)

    def forward(self, x):
        # Pass the input through the layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation function here as we will apply it during loss computation
        return x


game = utility()
game.runGame(-1)  # positive value to execute a specific number of moves, set to -1 to play a full game

# game = utility()
# game.driver.get("https://play2048.co/")
# time.sleep(3)

# for __ in range(10):
#     game.getGameState()
#     game.printGameGrid()
#     move = game.generateRandomMove()
#     game.sendMove(move)
#     #time.sleep(0.1) #let page update
