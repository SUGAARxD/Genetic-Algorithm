from genetic_algorithm import GeneticAlgorithm

GA = GeneticAlgorithm(population_size=10,
                      chromosome_length=18,
                      crossover_probability=0.5,
                      mutation_probability=0.05)
GA.run(epochs=50)
