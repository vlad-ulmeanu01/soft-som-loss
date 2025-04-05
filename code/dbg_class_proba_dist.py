# cum arata distributia claselor de probabilitati peste SOM? pentru 10 batchuri.

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import json

import som_loss
import design
import loader
import utils

SOM_NET_ID = "1743675705"
SOM_MAP_LENGTH = 200
EPOCH_CHECKPOINT = 0 # 10

torch.manual_seed(utils.DEFAULT_SEED)


def main():
    # we want to compute for each image in the val/ folder its som centroid (e.g. unit positions weighted by p_bmu).

    if utils.MODEL_TYPE == "hw":
        net = design.HwNetworkGlobal(len_output = len(utils.HT_DIR_CLASS))
    else:
        net = design.VGGUntrained(len_output = len(utils.HT_DIR_CLASS))

    if EPOCH_CHECKPOINT != 0:
        net.load_state_dict(torch.load(f"../net_saves/net_{SOM_NET_ID}_{EPOCH_CHECKPOINT}.pt", weights_only = True, map_location = utils.DEVICE))
    net.eval()

    som = som_loss.SoftSomLoss2d(
        map_length = SOM_MAP_LENGTH,
        vector_length = net.fc_last_layer.in_features if utils.MODEL_TYPE == "hw" else net.classifier[-1].in_features,
        num_classes = len(utils.HT_DIR_CLASS),
        smoothing_kernel_std = 5
    )
    if EPOCH_CHECKPOINT != 0:
        som.weights = torch.load(f"../net_saves/som_weights_{SOM_NET_ID}_{EPOCH_CHECKPOINT}.pt", weights_only = True, map_location = utils.DEVICE)

    dsets = {}
    # # dsets["train"] = loader.Dataset(dset_type = "train")
    # dsets["test"] = loader.Dataset(dset_type = "test") # , class_ht = dsets["train"].class_ht
    # # del dsets["train"]

    dsets["train"] = loader.Dataset(dset_type = "train", break_after_fill_ht = True)
    dsets["test"] = loader.Dataset(dset_type = "test", class_ht = dsets["train"].class_ht)
    del dsets["train"]

    gens = {}
    gens["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size = 128, shuffle = True)

    for loop_id, (x, yTruth_indexes) in zip(range(1, 1+1), gens["test"]):
        x, yTruth_indexes = x.to(utils.DEVICE), yTruth_indexes.to(utils.DEVICE)

        with torch.no_grad():
            _, y_sll, _ = net(x)

            for i in range(4):
                for j in range(i+1, 4):
                    print(f"dbg y_sll allclose 1e-3? {torch.allclose(y_sll[i], y_sll[j], 1e-3)} 1e-5? {torch.allclose(y_sll[i], y_sll[j], 1e-5)} 1e-9? {torch.allclose(y_sll[i], y_sll[j], 1e-9)}")

            # partial code copied from SOM below:

            # sim_scores[i, j] = similarity (unnormalized dot product) between the i-th vector from y_sll and the j-th SOM unit
            # (excluding class statistics, so only the first self.vector_length positions for each unit).
            sim_scores = y_sll @ som.weights[:, :som.vector_length].T

            for i in range(4):
                for j in range(i+1, 4):
                    print(f"dbg sim_scores allclose 1e-1? {torch.allclose(sim_scores[i], sim_scores[j], 1e-1)} 1e-2? {torch.allclose(sim_scores[i], sim_scores[j], 1e-2)} 1e-3? {torch.allclose(sim_scores[i], sim_scores[j], 1e-3)}")
                print(f"dbg sim_scores[{i}] min = {sim_scores[i].min()}, max = {sim_scores[i].max()}, mean = {sim_scores[i].mean()}")

            # p_bmu_presm[i, j] = probability that the i-th vector from y_sll chooses the j-th SOM unit as its BMU (pre-smoothing)
            p_bmu_presm = F.softmax(sim_scores, dim = 1)

            p_bmu_presm_np = p_bmu_presm.cpu().detach().numpy().reshape(-1, som.map_length, som.map_length)
            classes = yTruth_indexes.cpu().detach().numpy()
            for batch_id in range(min(4, p_bmu_presm.shape[0])):
                fig, ax = plt.subplots(1, figsize = (10, 4))

                print(f"{batch_id = }, min = {p_bmu_presm_np[batch_id].min()}, max = {p_bmu_presm_np[batch_id].max()}")

                ax.imshow(p_bmu_presm_np[batch_id])
                ax.set_title(f"p_bmu_presm for {batch_id = } (class id = {utils.LOADED_CLASS_NAMES[classes[batch_id]]})")
                ax.set_axis_off()

                fig.savefig(f"../pics/cam_pic_{SOM_NET_ID}/dbg_p_bmu_presm_ckpt_{EPOCH_CHECKPOINT}_{loop_id}_{batch_id}.png", bbox_inches = "tight")
                plt.close(fig)

            for i in range(4):
                for j in range(i+1, 4):
                    print(f"dbg p_bmu_presm {i = }, {j = }, allclose 1e-3? {np.allclose(p_bmu_presm_np[i], p_bmu_presm_np[j], 1e-3)} 1e-5? {np.allclose(p_bmu_presm_np[i], p_bmu_presm_np[j], 1e-5)} 1e-9? {np.allclose(p_bmu_presm_np[i], p_bmu_presm_np[j], 1e-10)}")

                fig, ax = plt.subplots(1, figsize = (10, 4))
                ax.stairs(*np.histogram(p_bmu_presm_np[i].flatten()), fill = True)
                fig.savefig(f"../pics/cam_pic_{SOM_NET_ID}/dbg_p_bmu_presm_hist_ckpt_{EPOCH_CHECKPOINT}_{loop_id}_{i}.png", bbox_inches = "tight")
                plt.close(fig)

            # apply the smoothing kernel (cross-correlation with padding).
            p_bmu = F.conv2d(
                p_bmu_presm.view(-1, som.map_length, som.map_length).unsqueeze(dim = 1), # add a dimension for in_channels = 1. -1 <=> batch_size.
                som.smoothing_kernel.unsqueeze(dim = 0).unsqueeze(dim = 0), # add two dimensions for out_channels = in_channels = 1.
                padding = 3 * som.smoothing_kernel_std
            ).view(-1, som.map_length ** 2)

            p_bmu_np = p_bmu.cpu().detach().numpy().reshape(-1, som.map_length, som.map_length)
            for batch_id in range(min(4, p_bmu.shape[0])):
                fig, ax = plt.subplots(1, figsize = (10, 4))

                print(f"{batch_id = }, min = {p_bmu_np[batch_id].min()}, max = {p_bmu_np[batch_id].max()}")

                ax.imshow(p_bmu_np[batch_id])
                ax.set_title(f"p_bmu for {batch_id = } (class id = {utils.LOADED_CLASS_NAMES[classes[batch_id]]})")
                ax.set_axis_off()

                fig.savefig(f"../pics/cam_pic_{SOM_NET_ID}/dbg_p_bmu_ckpt_{EPOCH_CHECKPOINT}_{loop_id}_{batch_id}.png", bbox_inches = "tight")
                plt.close(fig)
        
            # tensor of shape [self.map_length**2, self.num_classes]. represents the probability distribution that a unit picks some class.
            class_proba_dist_per_unit = F.softmax(
                torch.hstack([
                    p_bmu[yTruth_indexes == z].mean(dim = 0).view(-1, 1)
                    for z in range(som.num_classes)
                ]),
                dim = 1
            ).reshape(som.map_length, som.map_length, -1).cpu().detach().numpy()

            for z in range(som.num_classes):
                print(f"nr elem batch cu clasa {z}: {(yTruth_indexes == z).sum()}")

            print(f"{class_proba_dist_per_unit.shape = }")

            fig, ax = plt.subplots(2, 5, figsize = (15, 6))
            for z, (i, j) in zip(itertools.count(), itertools.product(range(ax.shape[0]), range(ax.shape[1]))):
                print(f"{z = }, min: {class_proba_dist_per_unit[:, :, z].min()}, max: {class_proba_dist_per_unit[:, :, z].max()}")

                ax[i, j].imshow(class_proba_dist_per_unit[:, :, z], vmin = 0, vmax = 1)
                ax[i, j].set_title(utils.LOADED_CLASS_NAMES[z])
                ax[i, j].set_axis_off()

            fig.savefig(f"../pics/cam_pic_{SOM_NET_ID}/dbg_class_proba_dist_ckpt_{EPOCH_CHECKPOINT}_{loop_id}.png", bbox_inches = "tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
