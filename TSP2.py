# Python3 implementation of the above approach
from random import randint
import numpy as np

INT_MAX = 2147483647
# Number of cities in TSP
V = 48

# Names of the cities
GENES = "ABCDE"

# Starting Node Value
START = 0

# Initial population size for the algorithm
POP_SIZE = 100


# Structure of a GNOME
# defines the path traversed
# by the salesman while the fitness value
# of the path is stored in an integer


class individual:
    def __init__(self) -> None:
        self.gnome = ""
        self.fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness


# Function to return a random number
# from start and end
def rand_num(start, end):
    return randint(start, end - 1)


# Function to check if the character
# has already occurred in the string
def repeat(s, ch):
    for i in range(len(s)):
        if s[i] == ch:
            return True

    return False


# Function to return a mutated GNOME
# Mutated GNOME is a string
# with a random interchange
# of two genes to create variation in species
def mutatedGene(gnome):
    gnome = list(gnome)
    while True:
        r = rand_num(1, V)
        r1 = rand_num(1, V)
        if r1 != r:
            temp = gnome[r]
            gnome[r] = gnome[r1]
            gnome[r1] = temp
            break
    return ''.join(gnome)


# Function to perform crossover between two parent individuals
def crossover(parent1, parent2, mp):
    # Select a random crossover point
    crossover_point = rand_num(1, V)

    # Create offspring by combining genetic information from parents
    child1_gnome = parent1.gnome[:crossover_point] + parent2.gnome[crossover_point:]
    child2_gnome = parent2.gnome[:crossover_point] + parent1.gnome[crossover_point:]

    # Create new offspring individuals
    child1 = individual()
    child1.gnome = child1_gnome
    child1.fitness = cal_fitness(child1.gnome, mp)

    child2 = individual()
    child2.gnome = child2_gnome
    child2.fitness = cal_fitness(child2.gnome, mp)

    return child1, child2


# Function to return a valid GNOME string
# required to create the population
def create_gnome():
    gnome = "0"
    while True:
        if len(gnome) == V:
            gnome += gnome[0]
            break

        temp = rand_num(1, V)
        if not repeat(gnome, chr(temp + 48)):
            gnome += chr(temp + 48)

    return gnome


# Function to return the fitness value of a gnome.
# The fitness value is the path length
# of the path represented by the GNOME.
def cal_fitness(gnome, mp):
    f = 0
    for i in range(len(gnome) - 1):
        if mp[ord(gnome[i]) - 48][ord(gnome[i + 1]) - 48] == INT_MAX:
            return INT_MAX
        f += mp[ord(gnome[i]) - 48][ord(gnome[i + 1]) - 48]

    return f


# Function to return the updated value
# of the cooling element.
def cooldown(temp):
    return (99.9 * temp) / 100


# Comparator for GNOME struct.
# def lessthan(individual t1,
#             individual t2)
# :
#     return t1.fitness < t2.fitness
# Function to perform Order Crossover (OX) between two parent individuals
def order_crossover(parent1, parent2, mp):
    # Select two random crossover points
    crossover_points = sorted([rand_num(1, V - 1), rand_num(1, V - 1)])

    # Create offspring with the genetic information between the crossover points copied from parent1
    child1_gnome = parent1.gnome[crossover_points[0]:crossover_points[1]]
    child2_gnome = parent2.gnome[crossover_points[0]:crossover_points[1]]

    # Fill the remaining positions in the offspring with genetic information from parent2
    remaining_parent2 = [gene for gene in parent2.gnome if gene not in child1_gnome]
    remaining_parent1 = [gene for gene in parent1.gnome if gene not in child2_gnome]

    index_child1 = 0
    index_child2 = 0

    for i in range(V):
        if len(child1_gnome) == V - crossover_points[1]:
            child1_gnome += ''.join(remaining_parent2[index_child1:])
            child2_gnome += ''.join(remaining_parent1[index_child2:])
            break

        if parent2.gnome[i] not in child1_gnome:
            child1_gnome = child1_gnome + parent2.gnome[i]
            index_child1 += 1

        if parent1.gnome[i] not in child2_gnome:
            child2_gnome = child2_gnome + parent1.gnome[i]
            index_child2 += 1

    # Create new offspring individuals
    child1 = individual()
    child1.gnome = ''.join(child1_gnome)
    child1.fitness = cal_fitness(child1.gnome, mp)

    child2 = individual()
    child2.gnome = ''.join(child2_gnome)
    child2.fitness = cal_fitness(child2.gnome, mp)

    return child1, child2


# Utility function for TSP problem.
def TSPUtil(mp):
    # Generation Number
    gen = 1
    # Number of Gene Iterations
    gen_thres = 5000

    population = []
    # temp = individual()
    best_score = INT_MAX

    # Populating the GNOME pool.
    for i in range(POP_SIZE):
        temp = individual()
        temp.gnome = create_gnome()
        temp.fitness = cal_fitness(temp.gnome, mp)
        population.append(temp)

    print("\nInitial population: \nGNOME  FITNESS VALUE\n")
    for i in range(POP_SIZE):
        print(population[i].gnome, population[i].fitness)
    print()

    found = False
    temperature = 10000

    # Iteration to perform
    # population crossing and gene mutation.
    while temperature > 500 and gen <= gen_thres:
        population.sort()
        print("\nCurrent temp: ", temperature)
        new_population = []

        # Perform crossover and mutation to create new population
        for i in range(POP_SIZE):

            while True:
                # Select two parent individuals randomly
                parent1 = population[rand_num(0, POP_SIZE)]
                parent2 = population[rand_num(0, POP_SIZE)]

                # Perform Order Crossover to create two children
                child1, child2 = order_crossover(parent1, parent2, mp)

                # Perform mutation on children
                child1.gnome = mutatedGene(child1.gnome)
                child2.gnome = mutatedGene(child2.gnome)

                better_child = child1 if child1.fitness < child2.fitness else child2

                if better_child.fitness <= parent1.fitness or better_child.fitness <= parent2.fitness:
                    new_population.append(better_child)
                    break

                else:

                    # Accepting the rejected children at
                    # a possible probability above threshold.
                    prob = pow(2.7, -1 * ((float)(better_child.fitness - population[i].fitness) / temperature))
                    if prob > 0.5:
                        new_population.append(better_child)
                        break

                # # Add children to new population
                # new_population.append(child1)
                # new_population.append(child2)

                best_score = min(best_score, better_child.fitness)

        print("BEST SCORE", best_score)
        temperature = cooldown(temperature)
        population = new_population
        print("Generation", gen)
        print("GNOME  FITNESS VALUE")

        # for i in range(POP_SIZE):
        #     print(population[i].gnome, population[i].fitness)
        gen += 1


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


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
    TSPUtil(adjacency_matrix)
