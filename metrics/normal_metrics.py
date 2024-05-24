import torch
from torch import Tensor
from torch.nn.functional import normalize, l1_loss
from typing import Optional


def validate_input_tensors(
    content_name: str, prediction: Tensor, target: Tensor, mask: Optional[Tensor] = None
):
    """Verify that the given tensors have the same shape. If they do not, leave a helpful error message."""
    if not (target.shape == prediction.shape):
        raise ValueError(
            f"{content_name} tensors are of different shapes ({target.shape} {prediction.shape})."
        )

    if mask is not None and target.shape != mask.shape:
        raise ValueError(
            f"{content_name} mask and {content_name} tensors have different shapes ({target.shape} and {mask.shape})."
        )


def RMS_error(
    prediction: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    content_name: str = "Input",
):
    """Root mean squared error.

    Input tensors should have the same shape.

    Args:
        prediction: Predicted values.
        target: Ground truth values.
        mask: Marks valid values with `1` and invalid ones with `0`.
            Invalid values have 0 loss. Set to `None` to disable masking.
        content_name: Describes the data contained int the input tensors for error logging.
            Error lines will look like `{contents_name} tensors have different shapes ({prediction_shape}
              and {target_shape})`
    """
    validate_input_tensors(content_name, prediction, target, mask)

    losses = (prediction - target) ** 2
    valid_pixel_count = losses.numel()
    if mask is not None:
        valid_pixel_count = mask.count_nonzero()
        losses *= mask

    return torch.sqrt(losses.sum() / valid_pixel_count)


def RMS_log_error(
    prediction: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    content_name: str = "",
):
    """Log-scaled root mean squared error.

    This computes the RMS error for log scaled versions of the input vectors.
    Input tensors should have the same shape.

    Args:
        prediction: Predicted values.
        target: Ground truth values.
        mask: Marks valid values with `1` and invalid ones with `0`.
            Invalid values have 0 loss. Set to `None` to disable masking.
        content_name: Describes the data contained int the input tensors for error logging.
        Error lines will look like `{contents_name} tensors have different shapes ({prediction_shape}
          and {target_shape})`
    """
    validate_input_tensors(content_name, prediction, target, mask)

    eps = 1e-6
    prediction = torch.clamp(prediction, min=eps)
    target = torch.clamp(target, min=eps)

    prediction = torch.log(prediction)
    target = torch.log(target)

    return RMS_error(prediction, target, mask)


def L1_relative_error(
    prediction: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    content_name: str = "",
) -> Tensor:
    """Mean of L1 losses scaled down by target values.

    'Relative' means that the errors are divided by the target values, as
    opposed to being absolute differences.
    Input tensors should have the same shape.

    Args:
        prediction: Predicted values.
        target: Ground truth values.
        mask: Marks valid values with `1` and invalid ones with `0`.
            Invalid values have 0 loss. Set to `None` to disable masking.
    """
    validate_input_tensors(content_name, prediction, target, mask)

    target = torch.clamp(target, min=1e-6)

    error = l1_loss(prediction, target, reduction="none") / target
    valid_pixel_count = error.numel()

    if mask is not None:
        valid_pixel_count = mask.count_nonzero()
        error *= mask
    return error.sum() / valid_pixel_count


def log10_error(
    prediction: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    content_name: str = "",
) -> Tensor:
    """Compute the L1 loss between the decimal logarithms of the inputs.

    Input tensors should have the same shape.

    Args:
        prediction: Predicted values.
        target: Ground truth values.
        mask: Marks valid values with `1` and invalid ones with `0`.
            Invalid values have 0 loss. Set to `None` to disable masking.
    """
    validate_input_tensors(content_name, prediction, target, mask)

    eps = 1e-6
    prediction = torch.clamp(prediction, min=eps)
    target = torch.clamp(target, min=eps)

    prediction = torch.log10(prediction)
    target = torch.log10(target)

    error = l1_loss(prediction, target, reduction="none")
    valid_pixel_count = error.numel()

    if mask is not None:
        valid_pixel_count = mask.count_nonzero()
        error *= mask
    return error.sum() / valid_pixel_count


def normals_angle_difference(
    prediction: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    content_name: str = "",
) -> Tensor:
    """Compute the angles between predicted and target normal vectors, in degrees.

    Input tensors should have the same shape.

    Args:
        prediction: Predicted values.
        target: Ground truth values.
        mask: Marks valid values with `1` and invalid ones with `0`.
            Invalid values have 0 loss. Set to `None` to disable masking.
    """
    validate_input_tensors(content_name, prediction, target, mask)

    prediction = normalize(prediction, dim=-1)
    target = normalize(target, dim=-1)

    cosines = torch.sum(prediction * target, dim=-1)

    # Make sure the interval is right to avoid `acos` creating NaN values
    cosines = torch.clamp(cosines, min=-1.0, max=1.0)

    radian_angles = torch.acos(cosines)
    degree_angles = torch.rad2deg(radian_angles)

    if mask is not None:
        # The mask has 3 channel dimensions, but the values in those channels are identical.
        # We need the mask's shape to match that of `degree_angles`.
        mask = mask[..., 0]
        degree_angles = degree_angles[mask.nonzero(as_tuple=True)]

    return degree_angles


def mean_of_values_under_threshold(
    values: Tensor, threshold: float, mask: Optional[Tensor] = None
) -> Tensor:
    if mask is not None:
        return torch.sum((values < threshold).logical_and(mask)) / mask.count_nonzero()

    return torch.sum(values < threshold) / values.numel()


def square_relative_error(
    prediction: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    content_name: str = "",
):
    """
    This computes the squared relative error between prediction and ground truth

    Args:
        prediction: Predicted values.
        target: Ground truth values.
        mask: Marks valid values with `1` and invalid ones with `0`.
            Invalid values have 0 loss. Set to `None` to disable masking.
    Returns:
        Squared relative error value
    """
    validate_input_tensors(content_name, prediction, target, mask)

    eps = 1e-6
    prediction = torch.clamp(prediction, min=eps)
    target = torch.clamp(target, min=eps)

    squared_rel = ((target - prediction) ** 2) / target
    valid_pixel_count = squared_rel.numel()

    if mask is not None:
        valid_pixel_count = mask.count_nonzero()
        squared_rel *= mask

    return squared_rel.sum() / valid_pixel_count


def delta_error(
    prediction: Tensor, target: Tensor, threshold: float, mask: Optional[Tensor] = None
):
    """
    Computes the point-wise relative difference between the prediction and the target values,
    then counts how many values in the result are under a given threshold, as a fraction of
    the total number of values.

    The point-wise ratio is the maximum of `prediction / target` and `target / prediction`.

    Args:
        prediction: Predicted values.
        target: Ground truth values.
        threshold: The threshold for counting ratios as passing the test.
        mask: Marks valid values with `1` and invalid ones with `0`.
            Set to `None` to disable masking.
    Returns:
        The ratio of computed values that fall under the threshold.
    """
    # Avoid dividing by zero
    eps = 1e-6
    prediction = torch.clamp(prediction, min=eps)
    target = torch.clamp(target, min=eps)

    ratios = torch.maximum((target / prediction), (prediction / target))

    if mask is not None:
        # Select only those ratios that the mask marks as valid.
        ratios = ratios[mask.nonzero(as_tuple=True)]

    res = mean_of_values_under_threshold(ratios, threshold)
    return res
