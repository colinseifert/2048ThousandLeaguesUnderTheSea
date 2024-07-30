import random
import numpy as np
import torch
from Game2048 import Game2048
from Game2048NN import Game2048NN

class GeneticAlgorithm2048:
    def __init__(self, population_size=10, generations=1000, mutation_probability=1 / 32):
        self.population_size = population_size
        self.generations = generations
        self.population = [self.create_individual() for _ in range(population_size)]
        self.mutation_probability = mutation_probability

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
        return game.get_score()

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

    def run(self):
        for generation in range(self.generations):
            scores = self.evaluate_population()
            print(f"Generation {generation + 1} scores: {scores}")

            for i, score in enumerate(scores):
                weights = self.get_weights(self.population[i])
                # print(f"Model {i + 1} weights: {weights}")

            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self.selection(scores)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population


if __name__ == "__main__":
    ga = GeneticAlgorithm2048()
    ga.run()
