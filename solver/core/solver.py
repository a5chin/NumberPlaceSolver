import copy
from typing import List


# To Be Imroved
class Solver():
    def __init__(self) -> None:
        self.data = None

    def get_result(self, data: List) -> List:
        self._set_num(data)

        return self.data

    def _set_num(self, data: List, idx: int = 0) -> bool:
        self.data = data
        if idx >= 81:
            return True

        col = idx % 9
        row = idx // 9
        if data[row][col] != 0:
            return self._set_num(data, idx + 1)

        line_data = {}
        for i in range(9):
            line_data[data[i][col]] = True
            line_data[data[row][i]] = True

        for val in range(1, 10):
            if val in line_data:
                continue
            if not self._check3x3(data, col, row, val):
                continue
            ndata = copy.deepcopy(data)
            ndata[row][col] = val
            if self._set_num(ndata, idx + 1):
                return True
        return False

    @staticmethod
    def _check3x3(data: List, col: int, row: int, val: int) -> bool:
        c3 = col // 3 * 3
        r3 = row // 3 * 3
        square_data = {}
        for x in range(3):
            for y in range(3):
                n = data[r3 + y][c3 + x]
                square_data[n] = True
        return val not in square_data
