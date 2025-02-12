import torch.nn.functional as F
import torch
import time

import utils

class SoftSomLoss2d:
    # callable from outside to schedule.
    def update_smoothing_kernel(self):
        # TODO: 2x kernele 1d in loc de unul 2d?
        self.smoothing_kernel = torch.exp(torch.distributions.normal.Normal(0, self.smoothing_kernel_std).log_prob(
            torch.tensor(
                [[(i**2 + j**2) ** 0.5 for j in torch.arange(-3*self.smoothing_kernel_std, 3*self.smoothing_kernel_std+1, 1)]
                                       for i in torch.arange(-3*self.smoothing_kernel_std, 3*self.smoothing_kernel_std+1, 1)], dtype = torch.float32, device = utils.DEVICE
            )
        ))
        self.smoothing_kernel /= self.smoothing_kernel.sum()


    def __init__(
        self,
        map_length: int,
        vector_length: int,
        num_classes: int,
        smoothing_kernel_std: float,
    ):
        self.map_length = map_length
        self.vector_length = vector_length
        self.num_classes = num_classes

        self.smoothing_kernel_std = int(smoothing_kernel_std)
        self.smoothing_kernel = None
        self.update_smoothing_kernel()

        # TODO: metode de initializare pentru self.weights? torch.nn.Parameter()?
        self.weights = torch.randn((self.map_length ** 2, self.vector_length + self.num_classes), requires_grad = True, device = utils.DEVICE)

        self.dbg_time_spent = {"other": 0.0, "loss_for": 0.0}


    """
    receives as input:
    * y_sll: the second to last layer of the net. its shape is [batch_size, self.vector_length].
    * classes: this is an int tensor with shape [batch_size].
    """
    def __call__(
        self,
        y_sll: torch.tensor,
        classes: torch.tensor
    ):
        dbg_time = time.time()

        # l2_dists[i, j] = L2 distance between the i-th vector from y_sll and the j-th SOM unit
        # (excluding class statistics, so only the first self.vector_length positions for each unit).
        l2_dists = torch.vstack([torch.linalg.vector_norm(y_sll[i] - self.weights[:, :self.vector_length], dim = 1) for i in range(len(y_sll))])

        # p_bmu_presm[i, j] = probability that the i-th vector from y_sll chooses the j-th SOM unit as its BMU (pre-smoothing)
        p_bmu_presm = F.softmin(l2_dists / l2_dists.sum(dim = 1).view(-1, 1), dim = 1)

        # apply the smoothing kernel (cross-correlation with padding).
        # also apply softmax over each unit (i.e. one softmax per each unit's batch_size probabilities. we want to force units to pick favourites)
        p_bmu = F.conv2d(
            p_bmu_presm.view(-1, self.map_length, self.map_length).unsqueeze(dim = 1), # add a dimension for in_channels = 1. -1 <=> batch_size.
            self.smoothing_kernel.unsqueeze(dim = 0).unsqueeze(dim = 0), # add two dimensions for out_channels = in_channels = 1.
            padding = 3 * self.smoothing_kernel_std
        ).view(-1, self.map_length ** 2)
        
        self.dbg_time_spent["other"] += time.time() - dbg_time; dbg_time = time.time()
       
        # tensor of shape [self.map_length**2, self.num_classes]. represents the probability distribution that a unit picks some class.
        class_proba_dist_per_unit = F.softmax(
            torch.hstack([
                p_bmu[classes == z].sum(dim = 0).view(-1, 1)
                for z in range(self.num_classes)
            ]),
            dim = 1
        )

        sm_weights = F.softmax(self.weights[:, self.vector_length:], dim = 1)
        log_sm_weights = torch.log(sm_weights + 1e-10)

        # the loss is the divergence between two distributions: unit class statistics, and the batch class distribution for that unit.
        # we also penalize if the the distributions unite by becoming uniform (mode collapse?).
        loss = F.kl_div(log_sm_weights, class_proba_dist_per_unit, reduction = "batchmean") -\
               (class_proba_dist_per_unit * torch.log(class_proba_dist_per_unit + 1e-10)).sum(dim = 1).mean() -\
               (sm_weights * log_sm_weights).sum(dim = 1).mean()

        self.dbg_time_spent["loss_for"] += time.time() - dbg_time

        return loss
