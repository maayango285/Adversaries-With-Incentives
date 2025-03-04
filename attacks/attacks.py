import numpy as np
import torch
import torch.nn as nn
from time import time


from attacks.base import Attack, LabelMixin
from attacks.util_fns import batch_clamp
from attacks.util_fns import batch_multiply
from attacks.util_fns import clamp
from attacks.util_fns import clamp_by_pnorm
from attacks.util_fns import is_float_or_torch_tensor
from attacks.util_fns import normalize_by_pnorm
from attacks.util_fns import rand_init_delta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def perturb_iterative(xvar, yvar, y2, predict, nb_iter, eps, eps_iter, loss_fn, delta_init=None, targeted=False, ord=np.inf, clip_min=0.0, clip_max=1.0, budget=None, multi_targets_mask=None, debug=False):
    """
    Iteratively maximize the loss over the input. It is a shared method for iterative attacks.
    Arguments:
        xvar (torch.Tensor): input data.
        yvar (torch.Tensor): target labels if targeted else input labels.
        y2 (torch.Tensor): input labels if targeted else None.
        predict (nn.Module): forward pass function.
        nb_iter (int): number of iterations.
        eps (float): maximum distortion.
        eps_iter (float): attack step size.
        loss_fn (nn.Module): loss function.
        delta_init (torch.Tensor): (optional) tensor contains the random initialization.
        targeted (bool): (optional) whether to minimize or maximize the loss.
        ord (int): (optional) the order of maximum distortion (inf or 2).
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
    Returns:
        torch.Tensor containing the perturbed input,
        torch.Tensor containing the perturbation
    """

    predict.eval()

    sum_grad = torch.zeros_like(yvar).float()

    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    if budget is None:
        budget = torch.ones_like(yvar)

    eps_iter *= budget
    eps *= budget
        
    delta.requires_grad_()
    for ii in range(nb_iter):
        outputs = predict(xvar + delta)
        
        if multi_targets_mask is not None: # loss is CE source \ source's targets
            assert targeted, 'expected a targeted attack'
            assert (loss_fn == 8), 'expected multi-targeted loss 8 (max target \ all)'
            
            # create mat containing targets scores
            t_outputs = torch.full(outputs.size(), -np.inf).type_as(outputs).to(device)
            t_outputs[multi_targets_mask] = outputs[multi_targets_mask] # fill targets logits
            
            # find max target score
            max_t_labels = t_outputs.argmax(1)
            # maximize max p(t) for t in T (loss num 8)
            loss = - nn.CrossEntropyLoss(reduction="sum")(outputs, max_t_labels) 
                
        else:
            assert (loss_fn == 0), 'invalid loss_fn value' # CE source \ all
            loss = nn.CrossEntropyLoss(reduction="sum")(outputs, yvar)
            if targeted:
                loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            curr_sum_grad = grad_sign.abs().sum(dim=[1,2,3])
            sum_grad += curr_sum_grad

            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)
        else:
            error = "Only ord=inf and ord=2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

    if debug:
        num_zero_grad = len(sum_grad[sum_grad == 0])
        print(f'num images with zero grad: {num_zero_grad}')
    x_adv = clamp(xvar + delta, clip_min, clip_max)
    r_adv = x_adv - xvar
    return x_adv, r_adv


class PGDAttack(Attack, LabelMixin):
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying within eps from the initial point.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
        eps_iter (float): attack step size.
        rand_init (bool): (optional) random initialization.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        ord (int): (optional) the order of maximum distortion (inf or 2).
        targeted (bool): if the attack is targeted.
        rand_init_type (str): (optional) random initialization type.
    """

    def __init__(
            self, predict, loss_fn=0, eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, targeted=False, rand_init_type='uniform'):
        super(PGDAttack, self).__init__(predict, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.rand_init_type = rand_init_type
        self.ord = ord
        self.targeted = targeted
        self.loss_fn = loss_fn
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None, y2=None, budget=None, multi_targets_mask=None, debug=False):
        """
        Given examples (x, y), returns their adversarial counterparts with an attack length of eps.
        Arguments:
            x (torch.Tensor): input tensor.
            y (torch.Tensor): label tensor.
                - if None and self.targeted=False, compute y as predicted
                labels.
                - if self.targeted=True, then y must be the targeted labels.
        Returns:
            torch.Tensor containing perturbed inputs,
            torch.Tensor containing the perturbation
        """
        # start = time()
        x, y, y2 = self._verify_and_process_inputs(x, y, y2)


        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            if self.rand_init_type == 'uniform':
                rand_init_delta(
                    delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
                delta.data = clamp(
                    x + delta.data, min=self.clip_min, max=self.clip_max) - x
            elif self.rand_init_type == 'normal':
                delta.data = 0.001 * torch.randn_like(x)  # initialize as in TRADES
            else:
                raise NotImplementedError(
                    'Only rand_init_type=normal and rand_init_type=uniform have been implemented.')


        x_adv, r_adv = perturb_iterative(
            x, y, y2, self.predict, nb_iter=self.nb_iter, eps=self.eps, eps_iter=self.eps_iter, loss_fn=self.loss_fn,
            targeted=self.targeted, ord=self.ord, clip_min=self.clip_min, clip_max=self.clip_max, delta_init=delta,
            budget=budget, multi_targets_mask=multi_targets_mask, debug=debug
        )
        # print(f'iter time {time() - start}')
        return x_adv.data, r_adv.data


class LinfPGDAttack(PGDAttack):
    """
    PGD Attack with order=Linf
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
        eps_iter (float): attack step size.
        rand_init (bool): (optional) random initialization.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        targeted (bool): if the attack is targeted.
        rand_init_type (str): (optional) random initialization type.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, rand_init_type='uniform'):
        ord = np.inf
        super(LinfPGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, rand_init=rand_init,
            clip_min=clip_min, clip_max=clip_max, targeted=targeted, ord=ord, rand_init_type=rand_init_type)


class L2PGDAttack(PGDAttack):
    """
    PGD Attack with order=L2
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
        eps_iter (float): attack step size.
        rand_init (bool): (optional) random initialization.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        targeted (bool): if the attack is targeted.
        rand_init_type (str): (optional) random initialization type.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, rand_init_type='uniform'):
        ord = 2
        super(L2PGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, rand_init=rand_init,
            clip_min=clip_min, clip_max=clip_max, targeted=targeted, ord=ord, rand_init_type=rand_init_type)
