import os
import random

import numpy as np
import torch
from Game2048 import Game2048
from Game2048NN import Game2048NN

class GeneticAlgorithm2048:
    def __init__(self, population_size=100, generations=5, mutation_probability=1 / 32, elitism_rate=0.1, model_dir='models'):
        self.population_size = population_size
        self.generations = generations
        self.mutation_probability = mutation_probability
        self.elitism_rate = elitism_rate
        self.population = [self.create_individual() for _ in range(population_size)]
        self.best_model = None
        self.best_score = -1
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.current_generation = self.load_population()

    def create_individual(self):
        model = Game2048NN()
        return model

    def get_weights(self, model):
        return model.fc1.weight.data.numpy().flatten()

    def set_weights(self, model, weights):
        model.fc1.weight.data = torch.tensor(weights.reshape((4, 16)), dtype=torch.float32)

    def fitness(self, model):
        game = Game2048()
        game.model = model
        game.run_game()
        # Improved fitness evaluation
        score = game.get_score()
        highest_tile = np.max(game.board)
        fitness_score = score + highest_tile

        # Track the best model
        if fitness_score > self.best_score:
            self.best_score = fitness_score
            self.best_model = model

        return fitness_score

    def selection(self, scores):
        total_score = sum(scores)
        selection_probs = [score / total_score for score in scores]
        selected_indices = np.random.choice(self.population_size, 2, p=selection_probs, replace=False)
        return self.population[selected_indices[0]], self.population[selected_indices[1]]

    def crossover(self, parent1, parent2):
        parent1_weights = self.get_weights(parent1)
        parent2_weights = self.get_weights(parent2)
        crossover_point = len(parent1_weights) // 2
        child_weights = np.concatenate((parent1_weights[:crossover_point], parent2_weights[crossover_point:]))
        child = self.create_individual()
        self.set_weights(child, child_weights)
        return child

    def mutate(self, model):
        weights = self.get_weights(model)
        for i in range(len(weights)):
            if random.random() < self.mutation_probability:
                weights[i] = random.uniform(-1, 1)
        self.set_weights(model, weights)

    def evaluate_population(self):
        scores = []
        for model in self.population:
            score = self.fitness(model)
            scores.append(score)
        return scores

    def save_population(self, generation):
        for i, model in enumerate(self.population):
            model_path = os.path.join(self.model_dir, f"model_gen{generation}_ind{i}.pt")
            model.save_model(model_path)

    def load_population(self):
        existing_files = [f for f in os.listdir(self.model_dir) if f.startswith('model_gen')]
        if not existing_files:
            return 0  # Start from generation 0 if no existing models

        # Find the highest generation number from the existing files
        max_gen = max(int(f.split('_')[1][3:]) for f in existing_files)
        for i in range(self.population_size):
            model_path = os.path.join(self.model_dir, f"model_gen{max_gen}_ind{i}.pt")
            if os.path.exists(model_path):
                self.population[i].load_model(model_path)
        return max_gen + 1  # Continue from the next generation

    def run(self):
        for generation in range(self.current_generation, self.current_generation + self.generations):
            scores = self.evaluate_population()
            sorted_scores = sorted(scores, reverse=True)
            print(f"Generation {generation + 1} scores: {sorted_scores}")

            self.save_population(generation)

            # Elitism: Copy the top individuals to the new population
            num_elites = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(scores)[-num_elites:]
            elites = [self.population[i] for i in elite_indices]

            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent1, parent2 = self.selection(scores)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population

        # Print the final game board state of the highest score model
        if self.best_model:
            best_game = Game2048()
            best_game.model = self.best_model
            best_game.run_game()
            print("Final board state of the highest score model:")
            best_game.print_board()
            print(f"Highest fitness score: {self.best_score}")
            # highest actual score on board
            print(f"Highest score: {best_game.get_score()}")


if __name__ == "__main__":
    ga = GeneticAlgorithm2048()
    ga.run()
