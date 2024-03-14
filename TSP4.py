import math

def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def greedy_tsp(coordinates, start=0):
    unvisited_cities = set(coordinates)
    current_city = coordinates[start]  # Start from the first city
    path = [current_city]

    unvisited_cities.remove(current_city)

    while unvisited_cities:
        nearest_city = min(unvisited_cities, key=lambda city: distance(current_city, city))
        path.append(nearest_city)
        unvisited_cities.remove(nearest_city)
        current_city = nearest_city

    path.append(coordinates[start])  # Return to the starting city to complete the cycle
    distance_traveled = sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))
    return distance_traveled


def nearest_neighbor_tsp(coordinates):
    num_cities = len(coordinates)
    unvisited_cities = set(range(num_cities))
    current_city = 0  # Start from the first city
    path = [current_city]

    unvisited_cities.remove(current_city)

    while unvisited_cities:
        nearest_city = min(unvisited_cities, key=lambda city: distance(coordinates[current_city], coordinates[city]))
        path.append(nearest_city)
        unvisited_cities.remove(nearest_city)
        current_city = nearest_city

    path.append(0)  # Return to the starting city to complete the cycle
    return path


def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def find_shortest_path(coordinates):
    n = len(coordinates)
    path = [0]
    remaining_points = set(range(1, n))

    while remaining_points:
        current_point = path[-1]
        next_point = min(remaining_points, key=lambda x: euclidean_distance(coordinates[current_point], coordinates[x]))
        path.append(next_point)
        remaining_points.remove(next_point)

    return path

def main():
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
    for i in range(len(coordinates)):
        shortest_path = greedy_tsp(coordinates, i)
        print("Starting City:", i)
        print("Total Distance:", shortest_path)

if __name__ == "__main__":
    main()




# for i in range(len(coordinates)):
#     shortest_path = greedy_tsp(coordinates, i)
#     # print("Shortest Path:", shortest_path)
#     print("Starting City:", i)
#     print("Total Distance:", sum(distance(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)))

# shortest_path = nearest_neighbor_tsp(coordinates)
# print("Shortest Path:", shortest_path)
# print("Total Distance:", sum(distance(coordinates[shortest_path[i]], coordinates[shortest_path[i + 1]]) for i in range(len(shortest_path) - 1)))


