import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import itertools
import torch
import os

import som_loss
import design
import loader
import utils


NORMAL_NET_ID = "1738760468"
SOM_NET_ID = "1738840750"
SOM_MAP_LENGTH = 200


def explain_cam(ims_classes, nets):
    num_channels = nets[0].conv_layers[-2].out_channels
    for i, (im, im_class) in zip(itertools.count(), ims_classes):
        fig, ax = plt.subplots(1, 3, figsize = (15, 15))

        wanted_class = utils.LOADED_CLASS_NAMES.index(im_class) # index of wanted class, here in [0, 9].

        ax[0].imshow(torch.permute(im[0].cpu(), (1, 2, 0)).detach().numpy())
        ax[0].set_title("Original image")

        for net, j, msg in zip(nets, range(1, 3), ["normal net", "net with SOM loss"]):
            # im.shape: [1, 3, 160, 160]
            out, out_sll, out_conv = net(im) # out_conv: torch.Size([1, 50, 33, 33]) -- 50 feature map outputs.

            sm = F.softmax(out.detach()[0], dim = 0)
            pred_class = utils.LOADED_CLASS_NAMES[sm.argmax().item()]

            print(f"out softmax = {torch.round(sm, decimals = 3)}, {pred_class = }", flush = True)

            # torch.Size([50, 1, 1]):
            kernel_weights = net.fc_last_layer.weight[wanted_class, net.fc_last_layer.in_features - num_channels:].unsqueeze(dim = -1).unsqueeze(dim = -1)

            cam = F.relu((out_conv * kernel_weights)[0]).sum(dim = 0)

            cam_im = ax[j].imshow(cam.cpu().detach().numpy())
            fig.colorbar(cam_im, ax = ax[j], orientation = "vertical", fraction = 0.046, pad = 0.04)
            ax[j].set_title(f"({msg}, predicted {pred_class} with\nproba {round(sm.max().item(), 2)}), CAM for {im_class} (proba {round(sm[wanted_class].item(), 2)})")
            ax[j].set_axis_off()

        fig.savefig(f"../pics/cam_pic_{SOM_NET_ID}/cam_pic_{i}.png", bbox_inches = "tight")
        plt.close(fig)


def explain_som_class_representation(som):
    order = som.weights.detach().cpu()[:, -som.num_classes:].argsort(dim = 1).reshape(som.map_length, som.map_length, -1)
    w = F.softmax(som.weights.detach().cpu()[:, -som.num_classes:], dim = 1).reshape(som.map_length, som.map_length, -1)

    try:
        fig, ax = plt.subplots(2, 5, figsize = (15, 6))
        for z, (i, j) in zip(itertools.count(), itertools.product(range(ax.shape[0]), range(ax.shape[1]))):
            ax[i, j].imshow((order[:, :, -1] == z) * 1.0 + (order[:, :, -2] == z) * 0.5, vmin = 0, vmax = 1)
            ax[i, j].set_title(utils.LOADED_CLASS_NAMES[z])
            ax[i, j].set_axis_off()

        fig.savefig(f"../pics/cam_pic_{SOM_NET_ID}/som_class_representation_arg_top2.png", bbox_inches = "tight")
        plt.close(fig)
    except:
        print("som class representation arg top2 failed.")

    try:
        fig, ax = plt.subplots(2, 5, figsize = (15, 6))
        for z, (i, j) in zip(itertools.count(), itertools.product(range(ax.shape[0]), range(ax.shape[1]))):
            ax[i, j].imshow(w[:, :, z], vmin = 0, vmax = 1)
            ax[i, j].set_title(utils.LOADED_CLASS_NAMES[z])
            ax[i, j].set_axis_off()

        fig.savefig(f"../pics/cam_pic_{SOM_NET_ID}/som_class_representation.png", bbox_inches = "tight")
        plt.close(fig)
    except:
        print("som class representation failed.")


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
    for fpath in [f"../net_saves/net_{NORMAL_NET_ID}_40.pt", f"../net_saves/net_{SOM_NET_ID}_40.pt"]:
        nets.append(design.HwNetworkGlobal(len_output = len(utils.HT_DIR_CLASS)))
        nets[-1].load_state_dict(torch.load(fpath, weights_only = True, map_location = utils.DEVICE))
        nets[-1].eval()

    som = som_loss.SoftSomLoss2d(map_length = SOM_MAP_LENGTH, vector_length = nets[0].fc_last_layer.in_features, num_classes = len(utils.HT_DIR_CLASS), smoothing_kernel_std = 2)
    som.weights = torch.load(f"../net_saves/som_weights_{SOM_NET_ID}_40.pt", weights_only = True, map_location = utils.DEVICE)

    print("Loaded nets, som weights.", flush = True)

    os.mkdir(f"../pics/cam_pic_{SOM_NET_ID}/")

    explain_cam(ims_classes, nets)
    explain_som_class_representation(som)


if __name__ == "__main__":
    main()
