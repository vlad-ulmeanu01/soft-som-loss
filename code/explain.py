import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import itertools
import torch

import som_loss
import design
import loader
import utils


def explain_cam(ims_classes, nets):
    num_channels = nets[0].conv_layers[-2].out_channels
    for i, (im, im_class) in zip(itertools.count(), ims_classes):
        fig, ax = plt.subplots(1, 3, figsize = (15, 15))

        wanted_class = utils.LOADED_CLASS_NAMES.index(im_class) # index of wanted class, here in [0, 9].

        ax[0].imshow(torch.permute(im[0].cpu(), (1, 2, 0)).detach().numpy())
        ax[0].set_title("Original image")

        for net, msg, j in zip(nets, ["normal net", "net with SOM loss"], range(1, 3)):
            # im.shape: [1, 3, 160, 160]
            out, out_sll, out_conv = net(im) # out_conv: torch.Size([1, 50, 33, 33]) -- 50 feature map outputs.

            sm = F.softmax(out.detach()[0], dim = 0)
            pred_class = utils.LOADED_CLASS_NAMES[sm.argmax().item()]

            print(f"{j = }, out softmax = {torch.round(sm, decimals = 3)}, {pred_class = }", flush = True)

            # torch.Size([50, 1, 1]):
            kernel_weights = net.fc_last_layer.weight[wanted_class, net.fc_last_layer.in_features - num_channels:].unsqueeze(dim = -1).unsqueeze(dim = -1)

            cam = F.relu((out_conv * kernel_weights)[0]).mean(dim = 0)

            ax[j].imshow(cam.cpu().detach().numpy())
            ax[j].set_title(f"({msg}, predicted {pred_class} with\nproba {round(sm.max().item(), 2)}), CAM for {im_class} (proba {round(sm[wanted_class].item(), 2)})")

        fig.savefig(f"../pics/cam_pic_{i}.png", bbox_inches = "tight")
        plt.close(fig)


def explain_som_class_representation(som):
    order = som.weights.detach().cpu()[:, -som.num_classes:].argsort(dim = 1).reshape(som.map_length, som.map_length, -1)

    fig, ax = plt.subplots(2, 5, figsize = (15, 6))
    for z, (i, j) in zip(itertools.count(), itertools.product(range(ax.shape[0]), range(ax.shape[1]))):
        ax[i, j].imshow((order[:, :, z] == 0) * 1.0 + (order[:, :, z] == 1) * 0.5, vmin = 0, vmax = 1)
        ax[i, j].set_title(utils.LOADED_CLASS_NAMES[z])
        ax[i, j].set_axis_off()

    fig.savefig(f"../pics/som_class_representation.png", bbox_inches = "tight")
    plt.close(fig)

    # w = F.softmax(som.weights.detach().cpu()[:, -som.num_classes:], dim = 1).reshape(som.map_length, som.map_length, -1)

    # fig, ax = plt.subplots(2, 5, figsize = (15, 6))
    # for z, (i, j) in zip(itertools.count(), itertools.product(range(ax.shape[0]), range(ax.shape[1]))):
    #     ax[i, j].imshow(w[:, :, z], vmin = 0, vmax = 1)
    #     ax[i, j].set_title(utils.LOADED_CLASS_NAMES[z])
    #     ax[i, j].set_axis_off()

    # fig.savefig(f"../pics/som_class_representation.png", bbox_inches = "tight")
    # plt.close(fig)


def main():
    # (image path, class for which I'd like to see a heatmap explanation).
    # additional comment per line where the wanted class equals the correct class.
    fpaths_classes = [
        ("../imagenette2-160/val/n01440764/ILSVRC2012_val_00029930.JPEG", "fish"), # fish (easy).
        ("../imagenette2-160/val/n01440764/ILSVRC2012_val_00029930.JPEG", "dog"),
        ("../imagenette2-160/val/n01440764/ILSVRC2012_val_00029930.JPEG", "golf_ball"),
        ("../imagenette2-160/val/n01440764/n01440764_141.JPEG", "fish"), # fish.
        ("../imagenette2-160/val/n01440764/n01440764_141.JPEG", "dog"),
        ("../imagenette2-160/val/n01440764/n01440764_141.JPEG", "golf_ball"),
        ("../imagenette2-160/val/n03000684/n03000684_32.JPEG", "chain_saw"), # chain_saw.
        ("../imagenette2-160/val/n03000684/n03000684_32.JPEG", "golf_ball"),
        ("../imagenette2-160/val/n03000684/n03000684_32.JPEG", "parachute"),
        ("../imagenette2-160/val/n03000684/n03000684_82.JPEG", "chain_saw"), # chain_saw.
        ("../imagenette2-160/val/n03000684/n03000684_82.JPEG", "golf_ball"),
        ("../imagenette2-160/val/n03000684/n03000684_82.JPEG", "parachute"),
        ("../imagenette2-160/val/n03417042/n03417042_3171.JPEG", "garbage_truck"), # garbage_truck
        ("../imagenette2-160/val/n03417042/n03417042_3171.JPEG", "chain_saw"),
        ("../imagenette2-160/val/n03417042/n03417042_3171.JPEG", "fish"),
        ("../imagenette2-160/val/n03417042/n03417042_3140.JPEG", "garbage_truck"), # garbage_truck
        ("../imagenette2-160/val/n03417042/n03417042_3140.JPEG", "parachute"),
        ("../imagenette2-160/val/n03888257/n03888257_482.JPEG", "parachute"), # parachute
        ("../imagenette2-160/val/n03888257/n03888257_482.JPEG", "garbage_truck")
    ]

    ims_classes = [
        (utils.get_px(fpath)[:, :utils.IM_LEN, :utils.IM_LEN].unsqueeze(dim = 0).to(utils.DEVICE), im_class) # add extra dim for batch size.
        for fpath, im_class in fpaths_classes
    ]

    nets = []
    for fpath in ["../net_saves/net_1731409474_100.pt", "../net_saves/net_1738590099_50.pt"]:
        nets.append(design.HwNetworkGlobal(len_output = len(utils.HT_DIR_CLASS)))
        nets[-1].load_state_dict(torch.load(fpath, weights_only = True, map_location = utils.DEVICE))
        nets[-1].eval()

    som = som_loss.SoftSomLoss2d(map_length = 50, vector_length = nets[0].fc_last_layer.in_features, num_classes = len(utils.HT_DIR_CLASS), smoothing_kernel_std = 1)
    som.weights = torch.load("../net_saves/som_weights_1738590099_50.pt", weights_only = True, map_location = utils.DEVICE)

    print("Loaded nets, som weights.", flush = True)

    # explain_cam(ims_classes, nets)
    explain_som_class_representation(som)


if __name__ == "__main__":
    main()
