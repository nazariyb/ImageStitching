import numpy as np
from math import sqrt, pi

__sobel_dx_matrix = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
__sobel_dy_matrix = __sobel_dx_matrix.transpose()


def __gradient_kernel(x, y, matrix):
    return matrix[x + 1, y + 1]


def __gradient_convolution(x0, y0, image, sobel_matrix):
    res = 0
    for i in range(x0 - 1, x0 + 2):
        for j in range(y0 - 1, y0 + 2):
            try:
                current_pixel = image[i, j]
                current_pixel = sqrt(current_pixel)
            except Exception:
                return 0
            res += __gradient_kernel(x0 - i, y0 - j,
                                     sobel_matrix) * current_pixel

    return res


def __gradient_execution(image, sobel_matrix):
    height = len(image)
    width = len(image[0])
    der = np.full((height, width), .0)

    for i in range(height):
        for j in range(width):
            der[i, j] = __gradient_convolution(i, j, image, sobel_matrix)

    return der


def dx(image):
    return __gradient_execution(image, __sobel_dx_matrix)


def dy(image):
    return __gradient_execution(image, __sobel_dy_matrix)


def dxx(image):
    return dx(dx(image))


def dyy(image):
    return dy(dy(image))


def dyx(image):
    return dx(dy(image))


def dxy(image):
    return dy(dx(image))


def magnitude(dx, dy):
    return np.sqrt(dx ** 2 + dy ** 2)


def angle(dx, dy):
    return (np.arctan2(dy, dx)) * 180 / pi
