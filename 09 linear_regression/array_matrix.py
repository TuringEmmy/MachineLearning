# author    TuringEmmy
# time      2018/11/11 15:15
# project   MachineLearning

"""
数组矩阵的练习
"""

import numpy as np


def array_operation():
    """
    数组运算
    :return:
    """
    a = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [2, 3, 7, 9]
    ]
    b = [
        2, 2, 2, 2
    ]

    result = np.multiply(a, b)
    print(result)
    return result


def matrix_operation():
    """
    矩阵运算
    :return:
    """
    a = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [2, 3, 7, 9]
    ]
    b = [
        [2], [2], [2], [2]
    ]

    result = np.dot(a, b)
    print(result)
    return result


if __name__ == '__main__':
    array_operation()
    matrix_operation()
