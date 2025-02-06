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
        p_bmu_presm = F.softmin(l2_dists / l2_dists.sum(dim = 1).view(-1, 1), dim = 1) # old: F.softmin(l2_dists, dim = 1).

        # apply the smoothing kernel (cross-correlation with padding).
        p_bmu = F.conv2d(
            p_bmu_presm.view(-1, self.map_length, self.map_length).unsqueeze(dim = 1), # add a dimension for in_channels = 1. -1 <=> batch_size.
            self.smoothing_kernel.unsqueeze(dim = 0).unsqueeze(dim = 0), # add two dimensions for out_channels = in_channels = 1.
            padding = 3 * self.smoothing_kernel_std
        ).view(-1, self.map_length ** 2)
        
        self.dbg_time_spent["other"] += time.time() - dbg_time; dbg_time = time.time()

        # TODO daca nu merge asta, incearca sa dai plot la cum se modifica in timp proba max de la softmin.
        sm_mat = F.softmax(self.weights[:, self.vector_length:], dim = 1)
        sm_argmax_cols = torch.argmax(sm_mat, dim = 1)
        sm_maxes = sm_mat[torch.arange(sm_mat.shape[0]), sm_argmax_cols]

        oh_mat_maxes_kept = torch.zeros_like(sm_mat)
        oh_mat_maxes_kept[torch.arange(oh_mat_maxes_kept.shape[0]), sm_argmax_cols] = sm_maxes

        # p_bmu[.] is an array of length map_length**2. instead of iterating through each batch element, we should instead add up all per-class contributions,
        # since the BCE is the same for two different batch samples from the same class.
        p_bmu_sum_per_class = [torch.zeros_like(p_bmu[0]) for z in range(self.num_classes)]
        for i in range(y_sll.shape[0]):
            p_bmu_sum_per_class[classes[i]] += p_bmu[i]

        loss = 0.0
        for z in range(self.num_classes):
            # instead of keeping the wanted distribution to 1 only on the classes[i] position, we instead keep the argmax position as well to its current value.
            # this way, we allow a small form of classes coexisting in the same unit. this will hopefully lead to clusters actually forming.
            oh_mat = oh_mat_maxes_kept.clone()
            oh_mat[:, z] = 1.0

            loss += (
                p_bmu_sum_per_class[z] * F.binary_cross_entropy(sm_mat, oh_mat, reduction = "none").mean(dim = 1)
            ).sum()
        loss /= y_sll.shape[0]

        self.dbg_time_spent["loss_for"] += time.time() - dbg_time

        return loss
