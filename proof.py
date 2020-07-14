import random
from math import sqrt

if __name__ == "__main__":

    sum_expected_path = 0
    sum_min_path = 0
    nloops = 1000000
    for i in range(nloops):
        x1 = random.random()
        y1 = random.random()
        x2 = random.random()
        y2 = random.random()
        x3 = random.random()
        y3 = random.random()

        # compute expected path length between 3 points (average of 3 possible paths)
        expected_path = 2/3 * (sqrt((x1-x2)**2 + (y1-y2)**2) +\
                sqrt((x2-x3)**2 + (y2-y3)**2) +\
                sqrt((x3-x1)**2 + (y3-y1)**2))
        sum_expected_path += expected_path

        # compute min path length
        path1 = sqrt((x1-x2)**2 + (y1-y2)**2) +\
                sqrt((x2-x3)**2 + (y2-y3)**2)       # 1, 2, 3
        path2 = sqrt((x2-x3)**2 + (y2-y3)**2) +\
                sqrt((x3-x1)**2 + (y3-y1)**2)       # 2, 3, 1
        path3 = sqrt((x3-x1)**2 + (y3-y1)**2) + \
                sqrt((x1-x2)**2 + (y1-y2)**2)       # 3, 1, 2
        min_path = min(path1, path2, path3)
        sum_min_path += min_path

    avg_expected_path = sum_expected_path / nloops
    avg_min_path = sum_min_path / nloops
    print(f"Avg Expected Path = {avg_expected_path}")
    print(f"Avg Min Path = {avg_min_path}")
