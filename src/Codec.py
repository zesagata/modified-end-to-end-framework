class Codec():
    def __init__(self, size, interpolate_size, mode):
        self.size = size
        self.interpolate_size = interpolate_size
        self.mode = mode

    def btc(self, image):
        if image.shape[0] % self.size != 0:
            n = ((self.size * int(image.shape[0] / self.size)) + self.size) - image.shape[0]
            image = numpy.pad(array=image, pad_width=(0, n))

        x = image.shape[0] / self.size
        y = image.shape[1] / self.size
        block_image = numpy.split(numpy.concatenate(numpy.split(image, y, axis=1)), x * y)
        for i in range(len(block_image)):
            mean = numpy.mean(numpy.mean(block_image[i], axis=1))
            std = numpy.std(block_image[i])
            m = self.size * self.size
            q = numpy.sum(block_image[i] > mean)

            a = mean - std * numpy.sqrt(q / (m - q))
            b = mean + std * numpy.sqrt((m - q) / q)

            block_image[i][block_image[i] > mean] = b
            block_image[i][block_image[i] < mean] = a

        block_image = numpy.concatenate(block_image)
        block_image = numpy.split(block_image, x)

        temp = numpy.concatenate(block_image, axis=1)

        if image.shape[0] % self.size != 0:
            temp = temp[1:-n + 1, 1:-n + 1]

        return torch.Tensor(temp)

    def BlockTruncationCoding(self, images_tensor):
        results_tensor = images_tensor.new_empty(size=images_tensor.size())
        for i in range(len(images_tensor)):
            compact_image = images_tensor[i][0]
            compressed_image = self.btc(compact_image)
            results_tensor[i][0] = compressed_image
        return results_tensor

    def Interpolate(self, image):
        return nn.functional.interpolate(input=image, scale_factor=self.interpolate_size, mode=self.mode,
                                         align_corners=False)
