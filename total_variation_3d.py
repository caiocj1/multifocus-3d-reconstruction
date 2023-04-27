import torch
from torch import Tensor

class TotalVariationL1(torch.nn.Module):
    """Calculate the total variation for one or batch tensor.

    The total variation is the sum of the absolute differences for neighboring
    voxel-values in the input multi-focus images.
    """

    def __init__(self, *, is_mean_reduction: bool = False) -> None:
        """Constructor.

        Args:
            is_mean_reduction (bool, optional):
                When `is_mean_reduction` is True, the sum of the output will be
                divided by the number of elements those used
                for total variation calculation. Defaults to False.
        """
        super(TotalVariationL1, self).__init__()
        self._is_mean = is_mean_reduction

    def forward(self, tensor_: Tensor) -> Tensor:
        return self._total_variation_l1(tensor_)

    def _total_variation_l1(self, tensor_: Tensor) -> Tensor:
        """Calculate total variation regularization norm for 3D reconstruction.

        Args:
            tensor_ (Tensor): input tensor must be the any following shapes:
                - 3-dimensional: [depth, height, width]
                - 4-dimensional: [channel, depth,  height, width]
                - 5-dimensional: [batch, channel, depth, height, width]

        Raises:
            ValueError: Input tensor is not either 3, 4 or 5-dimensional.

        Returns:
            Tensor: the output tensor shape depends on the size of the input.
                - Input tensor was 3 or 4 dimensional
                    return tensor as a scalar
                - Input tensor was 5 dimensional
                    return tensor as an array
        """
        ndims_ = tensor_.dim()

        if ndims_ == 3:
            z_diff = tensor_[1:, :, :] - tensor_[:-1, :, :]
            y_diff = tensor_[:, 1:, :] - tensor_[:, :-1, :]
            x_diff = tensor_[:, :, 1:] - tensor_[:, :, :-1]
        elif ndims_ == 4:
            z_diff = tensor_[:, 1:, :, :] - tensor_[:, :-1, :, :]
            y_diff = tensor_[:, :, 1:, :] - tensor_[:, :, :-1, :]
            x_diff = tensor_[:, :, :, 1:] - tensor_[:, :, :, :-1]
        elif ndims_ == 5:
            z_diff = tensor_[:, :, 1:, :, :] - tensor_[:, :, :-1, :, :]
            y_diff = tensor_[:, :, :, 1:, :] - tensor_[:, :, :, :-1, :]
            x_diff = tensor_[:, :, :, :, 1:] - tensor_[:, :, :, :, :-1]
        else:
            raise ValueError(
                'Input tensor must be either 3, 4 or 5-dimensional.')

        sum_axis = tuple({abs(x) for x in range(ndims_ - 4, ndims_)})
        z_denominator = (
            z_diff.shape[sum_axis[0]::].numel() if self._is_mean else 1
        )
        y_denominator = (
            y_diff.shape[sum_axis[0]::].numel() if self._is_mean else 1
        )
        x_denominator = (
            x_diff.shape[sum_axis[0]::].numel() if self._is_mean else 1
        )

        return (
            torch.sum(torch.abs(z_diff), dim=sum_axis) / z_denominator
            + torch.sum(torch.abs(y_diff), dim=sum_axis) / y_denominator
            + torch.sum(torch.abs(x_diff), dim=sum_axis) / x_denominator
        )

class TotalVariationL2(torch.nn.Module):
    """Calculate the total variation for one or batch tensor.

    The total variation is the sum of the absolute differences for neighboring
    voxel-values in the input multi-focus images.
    """

    def __init__(self, *, is_mean_reduction: bool = False) -> None:
        """Constructor.

        Args:
            is_mean_reduction (bool, optional):
                When `is_mean_reduction` is True, the sum of the output will be
                divided by the number of elements those used
                for total variation calculation. Defaults to False.
        """
        super(TotalVariationL2, self).__init__()
        self._is_mean = is_mean_reduction

    def forward(self, tensor_: Tensor) -> Tensor:
        return self._total_variation_l2(tensor_)

    def _total_variation_l2(self, tensor_: Tensor) -> Tensor:
        """Calculate total variation regularization norm for 3D reconstruction.

        Args:
            tensor_ (Tensor): input tensor must be the any following shapes:
                - 3-dimensional: [depth, height, width]
                - 4-dimensional: [channel, depth,  height, width]
                - 5-dimensional: [batch, channel, depth, height, width]

        Raises:
            ValueError: Input tensor is not either 3, 4 or 5-dimensional.

        Returns:
            Tensor: the output tensor shape depends on the size of the input.
                - Input tensor was 3 or 4 dimensional
                    return tensor as a scalar
                - Input tensor was 5 dimensional
                    return tensor as an array
        """
        ndims_ = tensor_.dim()

        if ndims_ == 3:
            z_diff = tensor_[1:, :, :] - tensor_[:-1, :, :]
            y_diff = tensor_[:, 1:, :] - tensor_[:, :-1, :]
            x_diff = tensor_[:, :, 1:] - tensor_[:, :, :-1]
        elif ndims_ == 4:
            z_diff = tensor_[:, 1:, :, :] - tensor_[:, :-1, :, :]
            y_diff = tensor_[:, :, 1:, :] - tensor_[:, :, :-1, :]
            x_diff = tensor_[:, :, :, 1:] - tensor_[:, :, :, :-1]
        elif ndims_ == 5:
            z_diff = tensor_[:, :, 1:, :, :] - tensor_[:, :, :-1, :, :]
            y_diff = tensor_[:, :, :, 1:, :] - tensor_[:, :, :, :-1, :]
            x_diff = tensor_[:, :, :, :, 1:] - tensor_[:, :, :, :, :-1]
        else:
            raise ValueError(
                'Input tensor must be either 3, 4 or 5-dimensional.')

        sum_axis = tuple({abs(x) for x in range(ndims_ - 4, ndims_)})
        z_denominator = (
            z_diff.shape[sum_axis[0]::].numel() if self._is_mean else 1
        )
        y_denominator = (
            y_diff.shape[sum_axis[0]::].numel() if self._is_mean else 1
        )
        x_denominator = (
            x_diff.shape[sum_axis[0]::].numel() if self._is_mean else 1
        )

        return (
            torch.sum(torch.pow(z_diff,2), dim=sum_axis) / z_denominator
            + torch.sum(torch.pow(y_diff,2), dim=sum_axis) / y_denominator
            + torch.sum(torch.pow(x_diff,2), dim=sum_axis) / x_denominator
        )        