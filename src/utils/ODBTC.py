import numpy as np
import torch


class ODBTC():
    def __init__(self, size):
        self.size = size
        self.ditherMatrix = self.bayer(size)
        max = np.max(self.ditherMatrix)
        min = np.min(self.ditherMatrix)
        self.ditherMatrix = (self.ditherMatrix - min) / (max - min)

    def rgb2grayscale(self, image):
        temp = image[0, :, :] * 0.2989 + image[1, :, :] * 0.587 + image[2, :, :] * 0.114
        temp = temp.astype(np.uint8)
        return temp

    def bayer(self, normalize=True):
        matrix = self.create_bayer(0, 0, self.size, 0, 1)
        return matrix / (self.size * self.size) \
            if normalize else matrix

    def create_bayer(self, x, y, size, value, step, matrix=None):
        if matrix is None:
            matrix = np.zeros((size, size))
        if (size == 1):
            matrix[y][x] = value
            return

        half = size // 2
        self.create_bayer(x, y, half, value + (step * 0), step * 4, matrix)
        self.create_bayer(x + half, y + half, half, value + (step * 1), step * 4, matrix)
        self.create_bayer(x + half, y, half, value + (step * 2), step * 4, matrix)
        self.create_bayer(x, y + half, half, value + (step * 3), step * 4, matrix)
        return matrix

    def __call__(self, image):
        grayscaleImage = self.rgb2grayscale(image)
        for i in range(len(image)):
            image[i] = self.filter(image[i], i, grayscaleImage)

        return image

    def filter(self, image_channel, c, gs):
        x = image_channel.shape[0] / self.size
        y = image_channel.shape[1] / self.size
        block_image = np.split(np.concatenate(np.split(image_channel, y, axis=1)), x * y)

        block_image_gs = np.split(np.concatenate(np.split(gs, y, axis=1)), x * y)

        for i in range(len(block_image)):
            min = np.min(block_image[i])
            max = np.max(block_image[i])

            gsMin = np.min(block_image_gs[i])

            k = max - min

            temp = self.ditherMatrix.copy()

            d = (temp * k) + gsMin

            block_image[i][block_image[i] >= d] = 1
            block_image[i][block_image[i] < d] = 0

        block_image = np.concatenate(block_image)
        block_image = np.split(block_image, x)

        temp_last = np.concatenate(block_image, axis=1)

        return temp_last
