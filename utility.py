import numpy
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

MOVES = ["UP", "DOWN", "LEFT", "RIGHT"]
gameGrid = numpy.zeros((4,4))


driver = webdriver.Chrome()
driver.get("https://play2048.co/")
row = 0
col = 0
selector = "//*[contains(@class, 'tile tile-')]"
elements = driver.find_elements(By.XPATH, selector)
print(elements)
for element in elements:
    # regex the string to get coordinates
    # convert text to int
    # update matrix with int at coordinates
    coordinates = re.search(r"tile-position-(\d+)-(\d+)", element.get_attribute('class'))
    row = int(coordinates.group(2)) -1
    col = int(coordinates.group(1)) -1
    gameGrid[row][col] = int(element.text)

print(gameGrid)




