import numpy
import re
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support  import expected_conditions as EC

UP = Keys.ARROW_UP
DOWN = Keys.ARROW_DOWN
LEFT = Keys.ARROW_LEFT
RIGHT = Keys.ARROW_RIGHT


class utility:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.gameGrid = numpy.zeros((4,4))
        return

    def runGame(self,numMoves):
        self.driver.get("https://play2048.co/")
        selector = "//*[contains(@class, 'tile tile-')]"
        waitCond = WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, selector))) #stored as a variable for debugging? 
        stopCondition = "body > div.container.upper > div.game-container > div.game-message.game-over"
        while(len(self.driver.find_elements(By.CSS_SELECTOR, stopCondition)) == 0 and numMoves != 0):
            waitCond = WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, selector)))
            move = self.generateRandomMove()
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

        # time.sleep(0.75) #wait for page to catch up after each move
        elements = self.driver.find_elements(By.XPATH, selector)
        #print(elements)
        for element in elements:
            # regex the string to get coordinates
            # convert text to int
            # update matrix with int at coordinates
            coordinates = re.search(r"tile-position-(\d+)-(\d+)", element.get_attribute('class'))
            row = int(coordinates.group(2)) -1 #decrement to have 0 indexing
            col = int(coordinates.group(1)) -1 # "  "   "   "   "   "   "
            self.gameGrid[row][col] = int(element.text)

    def generateRandomMove(self)->str:
        num = random.randint(1,4)
        #print("randomMove:",num)
        if(num == 1):
            return UP
        if(num == 2):
            return DOWN
        if(num == 3):
            return LEFT
        return RIGHT
    

    def sendMove(self, move):
        body = self.driver.find_element(By.TAG_NAME, 'body')
        body.send_keys(move)
        return
    
    def printGameGrid(self):
        print(self.gameGrid)
        return
    

game = utility()
game.runGame(-1) #positive value to execute a specific number of moves, set to -1 to play a full game

# game = utility()
# game.driver.get("https://play2048.co/")
# time.sleep(3)

# for __ in range(10):
#     game.getGameState()
#     game.printGameGrid()
#     move = game.generateRandomMove()
#     game.sendMove(move)
#     #time.sleep(0.1) #let page update





