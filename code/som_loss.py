import torch.nn.functional as F
import torch
import time

import utils

class SoftSomLoss2d:
    def __init__(
        self,
        map_length: int,
        vector_length: int,
        num_classes: int,
        lr: float,
        smoothing_kernel_std: float,
        p_bmu_thresh: float
    ):
        self.map_length = map_length
        self.vector_length = vector_length
        self.num_classes = num_classes
        self.lr = lr
        self.smoothing_kernel_std = int(smoothing_kernel_std)
        self.p_bmu_thresh = p_bmu_thresh

        # TODO: metode de initializare pentru self.weights? torch.nn.Parameter()?
        self.weights = torch.randn((self.map_length ** 2, self.vector_length + self.num_classes), requires_grad = True, device = utils.DEVICE)

        # TODO: 2x kernele 1d in loc de unul 2d?
        self.smoothing_kernel = torch.exp(torch.distributions.normal.Normal(0, self.smoothing_kernel_std).log_prob(
            torch.tensor(
                [[(i**2 + j**2) ** 0.5 for j in torch.arange(-2*self.smoothing_kernel_std, 2*self.smoothing_kernel_std+1, 1)]
                                       for i in torch.arange(-2*self.smoothing_kernel_std, 2*self.smoothing_kernel_std+1, 1)], dtype = torch.float32, device = utils.DEVICE
            )
        ))

        self.dbg_time_spent = {"l2_dists": 0.0, "p_bmu_presm": 0.0, "p_bmu": 0.0, "onehot_mat": 0.0, "loss_for": 0.0}

    # returns a float tensor with shape [num_classes]. It has 0 on all positions but wanted_class, where it has 1.
    def one_hot(self, wanted_class: int, num_classes: int):
        sol = torch.zeros(num_classes, device = utils.DEVICE)
        sol[wanted_class] = 1.0
        return sol

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

        self.dbg_time_spent["l2_dists"] += time.time() - dbg_time
        dbg_time = time.time()

        # p_bmu_presm[i, j] = probability that the i-th vector from y_sll chooses the j-th SOM unit as its BMU (pre-smoothing)
        p_bmu_presm = F.softmin(l2_dists, dim = 1)

        self.dbg_time_spent["p_bmu_presm"] += time.time() - dbg_time
        dbg_time = time.time()

        # apply the smoothing kernel (cross-correlation with padding).
        p_bmu = F.conv2d(
            p_bmu_presm.view(-1, self.map_length, self.map_length).unsqueeze(dim = 1), # add a dimension for in_channels = 1. -1 <=> batch_size.
            self.smoothing_kernel.unsqueeze(dim = 0).unsqueeze(dim = 0), # add two dimensions for out_channels = in_channels = 1.
            padding = 2 * self.smoothing_kernel_std
        ).view(-1, self.map_length ** 2)

        self.dbg_time_spent["p_bmu"] += time.time() - dbg_time
        dbg_time = time.time()

        onehot_mat = torch.vstack([self.one_hot(classes[i], self.num_classes) for i in range(y_sll.shape[0])])

        self.dbg_time_spent["onehot_mat"] += time.time() - dbg_time
        dbg_time = time.time()

        loss = 0.0
        # for i in range(y_sll.shape[0]):
        #     for j in range(self.map_length ** 2):
        #         if p_bmu[i, j] > self.p_bmu_thresh:
        #             loss += p_bmu[i, j] * F.binary_cross_entropy(
        #                 F.softmax(self.weights[j, self.vector_length:], dim = 0),
        #                 onehot_mat[i]
        #             )

        for i in range(y_sll.shape[0]):
            loss += (
                p_bmu[i] * F.binary_cross_entropy(
                    F.softmax(self.weights[:, self.vector_length:], dim = 1), # TODO softmax-ul asta e constant. pot sa-l calculez inainte si sa-l refolosesc aici?
                    onehot_mat[i].unsqueeze(dim = 0).broadcast_to(self.map_length ** 2, self.num_classes),
                    reduction = "none"
                ).mean(dim = 1)
            ).sum()
        loss /= y_sll.shape[0]

        self.dbg_time_spent["loss_for"] += time.time() - dbg_time

        return loss
