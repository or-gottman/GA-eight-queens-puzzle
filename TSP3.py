# Python3 implementation of the above approach
from random import randint, random
import numpy as np

INT_MAX = 2147483647
# Number of cities in TSP
V = 48

# Names of the cities
GENES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Initial population size for the algorithm
POP_SIZE = 100
ELITISM_PERCENTAGE = 0.1  # Percentage of elites to be preserved
MUTATION_RATE = 0.05  # Initial mutation rate
MIN_MUTATION_RATE = 0.01  # Minimum mutation rate
MAX_MUTATION_RATE = 0.1  # Maximum mutation rate
MUTATION_RATE_ADJUSTMENT = 0.1  # Mutation rate adjustment factor
MAX_GENERATIONS = 5000


# Structure of an individual
class Individual:
    def __init__(self):
        self.gene_sequence = ""
        self.fitness = 0


# Function to generate a random valid gene sequence
def create_gene_sequence():
    gene_sequence = "A"
    while len(gene_sequence) < V:
        temp = GENES[randint(0, len(GENES) - 1)]
        if temp not in gene_sequence:
            gene_sequence += temp
    gene_sequence += "A"  # Return to the starting node
    return gene_sequence


# Function to calculate the fitness value of an individual
def calculate_fitness(individual, distance_matrix):
    total_distance = 0
    for i in range(V):
        total_distance += distance_matrix[ord(individual.gene_sequence[i]) - 65][ord(individual.gene_sequence[i + 1]) - 65]
    individual.fitness = total_distance
    return total_distance


# Function to perform Order Crossover (OX) between two parent individuals
def order_crossover(parent1, parent2):
    crossover_points = sorted([randint(1, V - 1), randint(1, V - 1)])
    child1 = parent1.gene_sequence[crossover_points[0]:crossover_points[1]]
    child2 = parent2.gene_sequence[crossover_points[0]:crossover_points[1]]

    remaining_parent2 = [gene for gene in parent2.gene_sequence if gene not in child1]
    remaining_parent1 = [gene for gene in parent1.gene_sequence if gene not in child2]

    index_child1 = 0
    index_child2 = 0

    for i in range(V):
        if len(child1) == V - (crossover_points[1] - crossover_points[0]):
            child1 += ''.join(remaining_parent2[index_child1:])
            child2 += ''.join(remaining_parent1[index_child2:])
            break

        if parent2.gene_sequence[i] not in child1:
            child1 = child1 + parent2.gene_sequence[i]
            index_child1 += 1

        if parent1.gene_sequence[i] not in child2:
            child2 = child2 + parent1.gene_sequence[i]
            index_child2 += 1

    child1 = "A" + child1 + "A"  # Add start and end nodes
    child2 = "A" + child2 + "A"
    return child1, child2


# Function to perform mutation on an individual
def mutate(individual, mutation_rate):
    if random() < mutation_rate:
        mutated_index1 = randint(1, V)
        mutated_index2 = randint(1, V)
        individual.gene_sequence = (
            individual.gene_sequence[:mutated_index1]
            + individual.gene_sequence[mutated_index2]
            + individual.gene_sequence[mutated_index1 + 1 : mutated_index2]
            + individual.gene_sequence[mutated_index1]
            + individual.gene_sequence[mutated_index2 + 1 :]
        )


# Function to adjust the mutation rate based on the population diversity
def adjust_mutation_rate(population):
    max_fitness = max(individual.fitness for individual in population)
    min_fitness = min(individual.fitness for individual in population)
    diversity = max_fitness - min_fitness
    return max(MIN_MUTATION_RATE, min(MAX_MUTATION_RATE, diversity * MUTATION_RATE_ADJUSTMENT))


# Function to perform elitism
def elitism(population):
    elites_count = int(ELITISM_PERCENTAGE * len(population))
    elites = sorted(population, key=lambda x: x.fitness)[:elites_count]
    return elites


# Function to initialize the population
def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        new_individual = Individual()
        new_individual.gene_sequence = create_gene_sequence()
        population.append(new_individual)
    return population


# Main function for Genetic Algorithm
def genetic_algorithm(distance_matrix):
    population = initialize_population()
    generation = 0
    best_solution = None
    while generation < MAX_GENERATIONS:
        for individual in population:
            calculate_fitness(individual, distance_matrix)
        population.sort(key=lambda x: x.fitness)
        elites = elitism(population)
        if best_solution is None or population[0].fitness < best_solution.fitness:
            best_solution = population[0]

        new_population = elites[:]
        while len(new_population) < POP_SIZE:
            parent1, parent2 = population[randint(0, len(population) - 1)], population[randint(0, len(population) - 1)]
            child1, child2 = order_crossover(parent1, parent2)
            mutate(child1, MUTATION_RATE)
            mutate(child2, MUTATION_RATE)
            new_population.extend([child1, child2])

        MUTATION_RATE = adjust_mutation_rate(population)
        population = new_population
        generation += 1

    return best_solution


# Function to calculate the distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Function to create the adjacency matrix from the list of coordinates
def create_adjacency_matrix(coordinates):
    n = len(coordinates)
    adjacency_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = euclidean_distance(coordinates[i], coordinates[j])
                adjacency_matrix[i][j] = distance
    return adjacency_matrix


if __name__ == "__main__":
    coordinates = [
        (6734, 1453), (2233, 10), (5530, 1424), (401, 841), (3082, 1644),
        (7608, 4458), (7573, 3716), (7265, 1268), (6898, 1885), (1112, 2049),
        (5468, 2606), (5989, 2873), (4706, 2674), (4612, 2035), (6347, 2683),
        (6107, 669), (7611, 5184), (7462, 3590), (7732, 4723), (5900, 3561),
        (4483, 3369), (6101, 1110), (5199, 2182), (1633, 2809), (4307, 2322),
        (675, 1006), (7555, 4819), (7541, 3981), (3177, 756), (7352, 4506),
        (7545, 2801), (3245, 3305), (6426, 3173), (4608, 1198), (23, 2216),
        (7248, 3779), (7762, 4595), (7392, 2244), (3484, 2829), (6271, 2135),
        (4985, 140), (1916, 1569), (7280, 4899), (7509, 3239), (10, 2676),
        (6807, 2993), (5185, 3258), (3023, 1942)
    ]

    adjacency_matrix = create_adjacency_matrix(coordinates)
    best_solution = genetic_algorithm(adjacency_matrix)
    print("Best Solution:", best_solution.gene_sequence)
    print("Fitness:", best_solution.fitness)
