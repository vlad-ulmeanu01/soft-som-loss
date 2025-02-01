import torch

import utils

class SoftSomLoss2d:
    def __init__(
        self,
        map_length: int,
        vector_length: int,
        num_classes: int,
        lr: float,
        smoothing_kernel_std: float
    ):
        self.map_length = map_length
        self.vector_length = vector_length
        self.num_classes = num_classes
        self.lr = lr
        self.smoothing_kernel_std = int(smoothing_kernel_std)

        # TODO: metode de initializare pentru self.weights? torch.nn.Parameter()?
        self.weights = torch.randn((self.map_length ** 2, self.vector_length + self.num_classes), device = utils.DEVICE)

        # TODO: 2x kernele 1d in loc de unul 2d?
        self.smoothing_kernel = torch.exp(torch.distributions.normal.Normal(0, self.smoothing_kernel_std).log_prob(
            torch.tensor(
                [[(i**2 + j**2) ** 0.5 for j in torch.arange(-2*self.smoothing_kernel_std, 2*self.smoothing_kernel_std+1, 1)]
                                       for i in torch.arange(-2*self.smoothing_kernel_std, 2*self.smoothing_kernel_std+1, 1)], dtype = torch.float32, device = utils.DEVICE
            )
        ))

    # returns a float tensor with shape [num_classes]. It has -1 on all positions but wanted_class, where it has 1.
    def one_hot_tanh(self, wanted_class: int, num_classes: int):
        sol = -torch.ones(num_classes, device = utils.DEVICE)
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
        classes: torch.tensor,
    ):
        # l2_dists[i, j] = L2 distance between the i-th vector from y_sll and the j-th SOM unit
        # (excluding class statistics, so only the first self.vector_length positions for each unit).
        l2_dists = torch.vstack([torch.linalg.vector_norm(y_sll[i] - self.weights[:, :self.vector_length], dim = 1) for i in range(len(y_sll))])

        # p_bmu_presm[i, j] = probability that the i-th vector from y_sll chooses the j-th SOM unit as its BMU (pre-smoothing)
        p_bmu_presm = torch.nn.functional.softmin(l2_dists, dim = 1)

        # apply the smoothing kernel (cross-correlation with padding).
        p_bmu = torch.nn.functional.conv2d(
            p_bmu_presm.view(-1, self.map_length, self.map_length).unsqueeze(dim = 1), # add a dimension for in_channels = 1. -1 <=> batch_size.
            self.smoothing_kernel.unsqueeze(dim = 0).unsqueeze(dim = 0), # add two dimensions for out_channels = in_channels = 1.
            padding = 2 * self.smoothing_kernel_std
        ).view(-1, self.map_length ** 2)

        onehot_mat = torch.vstack([self.one_hot_tanh(classes[i], self.num_classes) for i in range(y_sll.shape[0])])

        with torch.no_grad():
            for i in range(y_sll.shape[0]):
                d = (self.lr * p_bmu[i]).unsqueeze(dim = -1).broadcast_to(self.weights.shape)
                self.weights = (1 - d) * self.weights + d * torch.cat([y_sll[i], onehot_mat[i]]).broadcast_to(self.weights.shape)

        loss = 0.0
        for i in range(y_sll.shape[0]):
            for j in range(self.map_length ** 2):
                loss += p_bmu[i, j] * torch.nn.functional.binary_cross_entropy(
                    (torch.nn.functional.tanh(self.weights[j, self.vector_length:]) + 1) * 0.5,
                    (onehot_mat[i] + 1) * 0.5
                )

        return loss
