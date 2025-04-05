import torch.nn.functional as F
import torch
import time

import utils

class SoftSomLoss2d:
    # callable from outside to schedule.
    def update_smoothing_kernel(self):
        if self.smoothing_kernel_std <= 0:
            return
        self.smoothing_kernel_1d = torch.exp(
            -(torch.arange(-3*self.smoothing_kernel_std, 3*self.smoothing_kernel_std + 1, 1) / (2 * self.smoothing_kernel_std)) ** 2
        ).to(utils.DEVICE)
        self.smoothing_kernel_1d /= self.smoothing_kernel_1d.sum()


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
        self.smoothing_kernel_1d = None
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

        # sim_scores[i, j] = similarity (normalized(?) dot product) between the i-th vector from y_sll and the j-th SOM unit
        # (excluding class statistics, so only the first self.vector_length positions for each unit).
        sim_scores = y_sll @ self.weights[:, :self.vector_length].T # * self.inv_sqrt_vector_len

        # p_bmu_presm[i, j] = probability that the i-th vector from y_sll chooses the j-th SOM unit as its BMU (pre-smoothing)
        p_bmu_presm = F.softmax(sim_scores, dim = 1)

        # apply the smoothing kernel (cross-correlation with padding).
        p_bmu = F.conv2d(
            F.conv2d(
                p_bmu_presm.view(-1, self.map_length, self.map_length).unsqueeze(dim = 1), # add a dimension for in_channels = 1. -1 <=> batch_size.
                self.smoothing_kernel_1d.view(1, 1, 1, -1),
                padding = "same"
            ),
            self.smoothing_kernel_1d.view(1, 1, -1, 1),
            padding = "same"
        ).view(-1, self.map_length ** 2)

        self.dbg_time_spent["other"] += time.time() - dbg_time; dbg_time = time.time()

        # tensor of shape [self.map_length**2, self.num_classes]. represents the probability distribution that a unit picks some class.
        class_proba_dist_per_unit = F.softmax(
            torch.hstack([
                p_bmu[classes == z].mean(dim = 0).view(-1, 1)
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
