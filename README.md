# 2048ThousandLeaguesUnderTheSea

This project uses a genetic algorithm to train a neural network to play the 2048 game.

## Prerequisites

- Python 3.8+
- pip (Python package installer)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/genetic-algorithm-2048.git
   cd genetic-algorithm-2048
   
2. Create and activate a python virtual environment
```bash
python -m venv myenv
# On Windows
myenv\Scripts\activate
# On macOS/Linux
source myenv/bin/activate
```

3. Install the required dependencies
```bash
pip install -r requirements.txt
```

## Running the Project

To run the genetic algorithm:
```bash
python GeneticAlgorithm2048.py
```

## Project Structure

**GeneticAlgorithm2048.py**: Entry point for running the genetic algorithm.

**MoveGenerator.py**: Contains the logic for generating moves based on the neural network.

**Game2048.py**: Implements the 2048 game mechanics.

**Game2048NN.py**: Defines the neural network model for the 2048 game.

**GameController.py**: Old code for running game in browser using selenium. **Do not use**

**requirements.txt**: Lists the required Python packages.