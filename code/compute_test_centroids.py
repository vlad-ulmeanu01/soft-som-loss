import torch.nn.functional as F
import itertools
import torch
import json

import som_loss
import design
import loader
import utils

NORMAL_NET_ID = "1739021122" # hw: 1738760468, vgg: 1739021122
SOM_NET_ID = "1739090246"
SOM_MAP_LENGTH = 200


def main():
    # we want to compute for each image in the val/ folder its som centroid (e.g. unit positions weighted by p_bmu).

    if utils.MODEL_TYPE == "hw":
        net = design.HwNetworkGlobal(len_output = len(utils.HT_DIR_CLASS))
    else:
        net = design.VGGUntrained(len_output = len(utils.HT_DIR_CLASS))

    net.load_state_dict(torch.load(f"../net_saves/net_{SOM_NET_ID}_40.pt", weights_only = True, map_location = utils.DEVICE))
    net.eval()

    som = som_loss.SoftSomLoss2d(
        map_length = SOM_MAP_LENGTH,
        vector_length = net.fc_last_layer.in_features if utils.MODEL_TYPE == "hw" else net.classifier[-1].in_features,
        num_classes = len(utils.HT_DIR_CLASS),
        smoothing_kernel_std = 2
    )
    som.weights = torch.load(f"../net_saves/som_weights_{SOM_NET_ID}_40.pt", weights_only = True, map_location = utils.DEVICE)

    dsets = {}
    dsets["train"] = loader.Dataset(dset_type = "train")
    dsets["test"] = loader.Dataset(dset_type = "test", class_ht = dsets["train"].class_ht)
    del dsets["train"]

    gens = {}
    gens["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size = 128)

    som_unit_locations = torch.tensor([(i, j) for i in range(som.map_length) for j in range(som.map_length)]).transpose(0, 1).to(utils.DEVICE)

    centroids, ys = [], []
    for loop_id, (x, yTruth_indexes) in zip(itertools.count(1), gens["test"]):
        x, yTruth_indexes = x.to(utils.DEVICE), yTruth_indexes.to(utils.DEVICE)

        _, y_sll, _ = net(x)

        # partial code copied from SOM below:
        with torch.no_grad():
            # l2_dists[i, j] = L2 distance between the i-th vector from y_sll and the j-th SOM unit
            # (excluding class statistics, so only the first self.vector_length positions for each unit).
            l2_dists = torch.vstack([torch.linalg.vector_norm(y_sll[i] - som.weights[:, :som.vector_length], dim = 1) for i in range(len(y_sll))])

            # p_bmu_presm[i, j] = probability that the i-th vector from y_sll chooses the j-th SOM unit as its BMU (pre-smoothing)
            p_bmu_presm = F.softmin(l2_dists / l2_dists.sum(dim = 1).view(-1, 1), dim = 1) # old: F.softmin(l2_dists, dim = 1).

            # apply the smoothing kernel (cross-correlation with padding).
            # also apply softmax over each unit (i.e. one softmax per each unit's batch_size probabilities. we want to force units to pick favourites)
            p_bmu = F.conv2d(
                p_bmu_presm.view(-1, self.map_length, self.map_length).unsqueeze(dim = 1), # add a dimension for in_channels = 1. -1 <=> batch_size.
                self.smoothing_kernel.unsqueeze(dim = 0).unsqueeze(dim = 0), # add two dimensions for out_channels = in_channels = 1.
                padding = 3 * self.smoothing_kernel_std
            ).view(-1, self.map_length ** 2)
            # p_bmu = F.softmax(p_bmu, dim = 0)
            # p_bmu /= p_bmu.sum(dim = 1).view(-1, 1)

        # first line contains centroid positions on Oi, second line on Oj.
        centroids.append(torch.vstack([(p_bmu * som_unit_locations[i]).sum(dim = 1) for i in range(2)]))
        ys.append(yTruth_indexes)

        del l2_dists, p_bmu_presm, p_bmu

        print(f"Finished {loop_id = }.", flush = True)

    centroids = torch.hstack(centroids).to("cpu")
    ys = torch.hstack(ys).to("cpu")

    ht = {i: [] for i in range(len(utils.HT_DIR_CLASS))} # class id -> [centroids].
    for j in range(len(ys)):
        ht[ys[j].item()].append((centroids[0, j].item(), centroids[1, j].item()))

    # we must load in the same order to get the images corresponding to the centroids.
    with open(f"../net_logs/test_images_centroids_net_som.json", "w") as fout:
        json.dump(ht, fout, indent = 4)


if __name__ == "__main__":
    main()
