from solver.core import Solver


def main():
    example = [
        [0, 9, 0, 6, 0, 1, 0, 2, 0],
        [8, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 3, 8, 4, 2, 5, 0, 0],
        [7, 0, 6, 0, 0, 0, 9, 0, 8],
        [0, 0, 1, 0, 5, 0, 7, 0, 0],
        [3, 0, 5, 0, 0, 0, 4, 0, 6],
        [0, 0, 9, 5, 1, 8, 6, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 2, 0, 4, 0, 8, 0],
    ]
    solver = Solver()
    result = solver.get_result(example)

    for res in result:
        print(*res)


if __name__ == "__main__":
    main()
