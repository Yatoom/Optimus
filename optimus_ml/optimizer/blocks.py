from scipy.spatial.distance import cityblock

obvserved = [
    [1, 2, 3, 4],
    [1, 2, 3, 5],
    [1, 2, 3, 6],
    [1, 2, 5, 9]
]

possibilities = [
    range(0, 5),
    range(0, 5),
    range(0, 8),
    range(0, 11)
]


def block_size(possibilities):
    return cityblock([x[0] for x in possibilities], [x[-1] for x in possibilities])


def score(observed, point, block_size):
    total = 0
    for observation in observed:
        total += cityblock(point, observation) / block_size

    return total / len(observed) ** 2
