import torch
from torch.nn.functional import avg_pool2d, max_pool2d_with_indices


def get_batch_conv_info(input_tensor, kernel_size=10, stride=5):
    # Set input tensor shape
    input_tensor_shape = input_tensor.shape
    if len(input_tensor_shape) == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif len(input_tensor_shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    elif len(input_tensor_shape) == 4:
        pass
    else:
        raise ValueError(f'Input tensor has invalid shape: {input_tensor_shape}')


    # Apply the pooling layer to the input tensor
    mean_tensor = avg_pool2d(input_tensor, kernel_size=kernel_size, stride=stride)

    # Compute the range tensor
    max_tensor, max_indices = max_pool2d_with_indices(input_tensor, kernel_size=kernel_size, stride=stride)
    min_tensor, min_indices = max_pool2d_with_indices(-input_tensor, kernel_size=kernel_size, stride=stride)
    min_tensor = -min_tensor
    range_tensor = max_tensor - min_tensor

    # Merge the mean and variance tensors as channels in a single tensor
    batch_info_tensor = torch.cat([mean_tensor, range_tensor, max_tensor, min_tensor, max_indices, min_indices], dim=1)

    # Return the batch info tensor
    return batch_info_tensor

