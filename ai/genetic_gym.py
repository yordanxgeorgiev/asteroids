import copy
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pickle

from ai.nn import DenseLayer, NeuralNetwork

SAVE_PATH = "./ai/best_ship.pkl"

MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2


def evaluate_ship(ship):
    from main import Game

    games = 3
    results = []

    for _ in range(games):

        game = Game(visual=False)

        frames, score = game.sim(ship)

        player = game.player

        fitness = (
            np.sqrt(frames) * 10
            + score * 20 * player.accuracy / 100
            + player.distance_travelled * 0.3
        )

        results.append(fitness)

    return float(np.mean(results))


class GeneticGym:
    """
    Genetic algorithm trainer for Asteroids neural networks.
    """

    _ELITES_COUNT = 2

    def __init__(self, population_size: int):
        self.gen_num = 0
        self.population_size = population_size
        self.population = [self._ship_factory() for _ in range(population_size)]

        self._mutation_rate: float | None = None
        self._mutation_strength: float | None = None

        self.pool = Pool(processes=cpu_count())

    @property
    def mutation_rate(self) -> float:
        if self._mutation_rate is None:
            self._mutation_rate = max(
                0.01,
                MUTATION_RATE * (0.998**self.gen_num),
            )

        return self._mutation_rate

    @property
    def mutation_strength(self) -> float:
        if self._mutation_strength is None:
            self._mutation_strength = max(
                0.01,
                MUTATION_STRENGTH * (0.995**self.gen_num),
            )

        return self._mutation_strength

    def _ship_factory(self) -> NeuralNetwork:

        from ai.nn import relu, sigmoid

        return NeuralNetwork(
            DenseLayer(36, 64, activation=relu, initializer="he"),
            DenseLayer(64, 32, activation=relu, initializer="he"),
            DenseLayer(32, 4, activation=sigmoid, initializer="xavier"),
        )

    def eval_population(self) -> list[float]:
        return self.pool.map(
            evaluate_ship,
            self.population,
        )

    def get_mating_pool(
        self,
        fitness_scores: list[float],
        tournament_k: int = 3,
    ) -> list[NeuralNetwork]:
        ranked = np.argsort(fitness_scores)[::-1]
        parents = [
            copy.deepcopy(self.population[i]) for i in ranked[: self._ELITES_COUNT]
        ]

        while len(parents) < self.population_size // 2:
            contenders = random.sample(
                range(self.population_size),
                tournament_k,
            )

            winner = max(
                contenders,
                key=lambda i: fitness_scores[i],
            )

            parents.append(copy.deepcopy(self.population[winner]))

        return parents

    def crossover(
        self,
        parent1: NeuralNetwork,
        parent2: NeuralNetwork,
    ) -> NeuralNetwork:

        child_layers = []

        for layer1, layer2 in zip(parent1.layers, parent2.layers):
            weight_mask = np.random.random(layer1.weights.shape) < 0.5
            weights = np.where(
                weight_mask,
                layer1.weights,
                layer2.weights,
            )
            bias_mask = np.random.random(layer1.biases.shape) < 0.5
            biases = np.where(
                bias_mask,
                layer1.biases,
                layer2.biases,
            )
            child_layers.append(
                DenseLayer(
                    input_dim=layer1.weights.shape[0],
                    output_dim=layer1.weights.shape[1],
                    activation=layer1.activation,
                    weights=weights.copy(),
                    biases=biases.copy(),
                    initializer="he",
                )
            )

        return NeuralNetwork(*child_layers)

    def mutate(self, network: NeuralNetwork):
        for layer in network.layers:
            weight_mask = np.random.random(layer.weights.shape) < self.mutation_rate
            layer.weights += (
                weight_mask
                * np.random.randn(*layer.weights.shape)
                * self.mutation_strength
            )

            bias_mask = np.random.random(layer.biases.shape) < self.mutation_rate
            layer.biases += (
                bias_mask
                * np.random.randn(*layer.biases.shape)
                * self.mutation_strength
            )

            if random.random() < 0.05:
                layer.weights += np.random.randn(*layer.weights.shape) * 0.5

    def next_generation(self) -> tuple[float, float]:
        fitness_scores = self.eval_population()
        parents = self.get_mating_pool(fitness_scores)

        new_population = [copy.deepcopy(x) for x in parents[: self._ELITES_COUNT]]

        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)

            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.gen_num += 1
        self._mutation_rate = None
        self._mutation_strength = None

        return (
            max(fitness_scores),
            float(np.mean(fitness_scores)),
        )

    def train(
        self,
        generations: int = 100,
        save_result: bool = False,
        display_champion: bool = True,
    ):

        try:
            for i in range(generations):
                best, average = self.next_generation()

                if i and i % 10 == 0:
                    print(
                        f"generation={self.gen_num} "
                        f"best={best:.2f} "
                        f"average={average:.2f}"
                    )

            fitness = self.eval_population()
            champion = self.population[int(np.argmax(fitness))]

            if save_result:
                with open(SAVE_PATH, "wb") as file:

                    pickle.dump(champion, file)

            if display_champion:
                from main import Game

                input("Training complete. Press ENTER to watch.")
                Game().start(ship_ai=champion)

        finally:
            self.pool.close()
            self.pool.join()


if __name__ == "__main__":

    GeneticGym(population_size=200).train(
        generations=1000,
        save_result=True,
    )
