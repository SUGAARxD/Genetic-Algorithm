import random

from fitness_function import evaluate_fitness
from chromosome import Chromosome


class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_length, crossover_probability, mutation_probability):
        Chromosome.length = chromosome_length
        self.population_size = population_size
        self.population = [Chromosome() for i in range(self.population_size)]
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability

    def run(self, epochs):
        file = open('output1.txt', 'w')
        for epoch in range(epochs + 1):
            file.write(f'#Generation {epoch} start\n')
            print(f'#Generation {epoch} start\n')
            i = 0
            for chromosome in self.population:
                file.flush()
                i += 1
                print(f'Chromosome {i}')

                evaluate_fitness(chromosome)

                file.write(f'Chromosome: n={chromosome.n}, lr={chromosome.lr}, wd={chromosome.wd}\n')
                file.write(f'Value: {chromosome.fitness_value}\n')

            if epoch != epochs:
                self.selection()
                self.crossover()
                self.mutation()

            file.write(f'#Generation {epoch} end\n\n')
            print(f'\n#Generation {epoch} end\n\n')

        file.close()

    def selection(self):
        total_fitness = sum(chromosome.fitness_value for chromosome in self.population)

        if total_fitness == 0:
            return

        cumulative_probability = 0

        for chromosome in self.population:
            chromosome.selection_probability = chromosome.fitness_value / total_fitness
            cumulative_probability += chromosome.selection_probability
            chromosome.cumulative_probability = cumulative_probability

        new_population = []
        for i in range(self.population_size):
            random_value = random.random()
            for chromosome in self.population:
                if random_value <= chromosome.cumulative_probability:
                    new_population.append(chromosome)
                    break

        self.population = new_population

    def crossover(self):
        parents = [self.population[i] for i in range(self.population_size) if
                   random.random() < self.crossover_probability]
        if len(parents) % 2 != 0:
            parents.pop()

        for i in range(0, len(parents), 2):
            crossover_point = random.randint((Chromosome.length // 6), Chromosome.length - (Chromosome.length // 6))

            child1_genes = parents[i].genes[:crossover_point] + parents[i + 1].genes[crossover_point:]
            child2_genes = parents[i + 1].genes[:crossover_point] + parents[i].genes[crossover_point:]

            parents[i].genes = child1_genes
            parents[i + 1].genes = child2_genes

    def mutation(self):
        for i in range(self.population_size):
            self.population[i].mutate(self.mutation_probability)
