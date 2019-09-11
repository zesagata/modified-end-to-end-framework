import torch.nn as nn


def loss_function_l1(reconstructed_image,original_image):
    return nn.MSELoss(size_average=False)(reconstructed_image,original_image)


def loss_function_l2(residual_image,decoded_image,original_image):
    return nn.MSELoss(size_average=False)(residual_image,original_image-decoded_image)