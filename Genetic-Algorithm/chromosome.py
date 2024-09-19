import random


class Chromosome:
    # it would be nice to keep the length divisible by 3 and not less than 6
    length = 18

    def __init__(self):
        self.genes = [random.randint(0, 1) for _ in range(Chromosome.length)]

        # n - number of neurons in the hidden layer
        # lr - learning rate for training
        # wd - weight decay for training
        self.n = 1
        self.lr = 1
        self.wd = 1

        self.fitness_value = 0
        self.selection_probability = 1
        self.cumulative_probability = 1

    def mutate(self, mutation_probability):
        for i in range(len(self.genes)):
            if random.random() < mutation_probability:
                self.genes[i] = 1 - self.genes[i]
