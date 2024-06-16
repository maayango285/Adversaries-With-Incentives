import torch
import torch.nn as nn

from attacks.util_fns import replicate_input


class Attack(object):
    """
    Abstract base class for all attack classes.
    Arguments:
        predict (nn.Module): forward pass function.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
    """

    def __init__(self, predict, clip_min, clip_max):
        self.predict = predict
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x, **kwargs):
        """
        Virtual method for generating the adversarial examples.
        Arguments:
            x (torch.Tensor): the model's input tensor.
            **kwargs: optional parameters used by child classes.
        Returns:
            adversarial examples.
        """
        error = "Sub-classes must implement perturb."
        raise NotImplementedError(error)

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)


class LabelMixin(object):
    def _verify_and_process_inputs(self, x, y, y_target=None):
        assert y is not None

        x = replicate_input(x)
        y = replicate_input(y)
        if y_target is not None:
            y_target = replicate_input(y_target)
        return x, y, y_target