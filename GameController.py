import re
import time

import numpy
import torch.nn.functional as F
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from Game2048NN import Game2048NN
from MoveGenerator import MoveGenerator


class GameController:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.gameGrid = numpy.zeros((4, 4))
        self.gameScore = 0
        self.model = Game2048NN()
        return

    def run_game(self, numMoves):
        self.driver.get("https://play2048.co/")
        selector = "//*[contains(@class, 'tile tile-')]"
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, selector)))  # stored as a variable for debugging?
        stopCondition = "body > div.container.upper > div.game-container > div.game-message.game-over"
        while (len(self.driver.find_elements(By.CSS_SELECTOR, stopCondition)) == 0 and numMoves != 0):
            WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, selector)))
            self.get_game_state()
            self.print_game_grid()
            try:
                move = MoveGenerator.generate_NN_moves(self.gameGrid, self.model)
            except Exception as e:
                print(e)
                break
            self.send_move(move)
            numMoves -= 1
        finalState = self.get_game_state()
        self.print_game_grid()
        self.get_game_score()
        self.print_game_score()
        self.driver.quit()
        return finalState

    def get_game_state(self):
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

    def get_game_score(self):
        time.sleep(1)
        selector = "/html/body/div[2]/div[1]/div/div[1]"
        element = self.driver.find_element(By.XPATH, selector)
        self.gameScore = int(element.text)

    def print_game_score(self):
        print(self.gameScore)

    def send_move(self, move):
        body = self.driver.find_element(By.TAG_NAME, 'body')
        body.send_keys(move)
        return

    def print_game_grid(self):
        print(self.gameGrid)
        return


# game = GameController()
# game.run_game(-1)  # positive value to execute a specific number of moves, set to -1 to play a full game

# game = GameController()
# game.driver.get("https://play2048.co/")
# time.sleep(3)

# for __ in range(10):
#     game.get_game_state()
#     game.print_game_grid()
#     move = game.generate_random_move(self.gameGrid)
#     game.send_move(move)
#     #time.sleep(0.1) #let page update
