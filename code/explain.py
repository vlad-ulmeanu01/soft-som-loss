import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision
import itertools
import torch
import os

import som_loss
import design
import loader
import utils


NORMAL_NET_ID = "1739021122" # hw: 1738760468, vgg: 1739021122
SOM_NET_ID = "1739379754"
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


def explain_gradcam(ims_classes, nets, som):
    im_len = 160 if utils.MODEL_TYPE == "hw" else 240
    resizer = torchvision.transforms.Resize((im_len, im_len), antialias = None)
    gradients = None

    def fwd_grad_hook(module, input, output):
        def bkw_grad_hook(g):
            nonlocal gradients
            gradients = g.detach()
        output.register_hook(bkw_grad_hook) # register_hook e hook apelat la backward. se aplica pe tensor!

    for net in nets:
        if utils.MODEL_TYPE == "hw":
            net.conv_layers[-1].register_forward_hook(fwd_grad_hook)
        else:
            net.avgpool.register_forward_hook(fwd_grad_hook)

    for i, (im, im_class) in zip(itertools.count(), ims_classes):
        fig, ax = plt.subplots(1, 3, figsize = (15, 15))

        wanted_class = utils.LOADED_CLASS_NAMES.index(im_class) # index of wanted class, here in [0, 9].

        ax[0].imshow(torch.permute(im[0].cpu(), (1, 2, 0)).detach().numpy())
        ax[0].set_title("Original image")

        if utils.MODEL_TYPE == "vgg": # im.shape: [1, 3, 160, 160] (hw) or [1, 3, 240, 240] (vgg)
            im = resizer(im)

        for net, j, msg in zip(nets, range(1, 3), ["normal net", "net with SOM loss"]):
            gradients = None
            out, out_sll, out_conv = net(im)

            sm = F.softmax(out.detach()[0], dim = 0)
            pred_class = utils.LOADED_CLASS_NAMES[sm.argmax().item()]

            yTruth_indexes = torch.tensor([wanted_class], device = utils.DEVICE)
            yTruth = F.one_hot(yTruth_indexes, num_classes = len(utils.HT_DIR_CLASS)).float()

            # loss = F.cross_entropy(out, yTruth) + som(out_sll, yTruth_indexes) if j == 2 else F.cross_entropy(out, yTruth)
            # loss.backward()
            out[0, wanted_class].backward()

            # print(f"{j = }, {gradients.shape = }, {gradients.requires_grad = }") # expected: [1, 512, 7, 7].

            weights = gradients.mean(dim = [-2, -1]) # weights.shape = [1, 512].
            gradcam = F.relu((out_conv.detach() * weights.view(1, -1, 1, 1)).mean(dim = 1)) # gradcam.shape = [1, 7, 7].

            max_gradcam_val = gradcam.max().item()
            gradcam /= max_gradcam_val
            gradcam = resizer(gradcam) # gradcam.shape = [1, im_len, im_len]

            ax[j].imshow(torch.permute(im[0] * gradcam, (1, 2, 0)).cpu().numpy())
            ax[j].set_title(f"({msg}, predicted {pred_class} with\nproba {round(sm.max().item(), 2)}), GradCAM for {im_class}\n(proba {round(sm[wanted_class].item(), 2)})\nmax gradcam value = {round(max_gradcam_val, 6)}")
            ax[j].set_axis_off()
            
        fig.savefig(f"../pics/cam_pic_{SOM_NET_ID}/gradcam_pic_{i}.png", bbox_inches = "tight")
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
        ("../imagenette2-160/val/n03888257/n03888257_482.JPEG", "garbage_truck"),

        ("../imagenette2-160/val/n02102040/n02102040_6800.JPEG", "dog"), # dog
        ("../imagenette2-160/val/n02102040/n02102040_6800.JPEG", "fish"),
        ("../imagenette2-160/val/n02102040/n02102040_6892.JPEG", "dog"), # dog
        ("../imagenette2-160/val/n02102040/n02102040_6892.JPEG", "fish"),
        ("../imagenette2-160/val/n02102040/n02102040_6892.JPEG", "golf_ball"),
        ("../imagenette2-160/val/n02102040/n02102040_1222.JPEG", "dog"), # dog
        ("../imagenette2-160/val/n02102040/n02102040_1222.JPEG", "parachute"),
        ("../imagenette2-160/val/n03445777/n03445777_15131.JPEG", "golf_ball"), # golf_ball
        ("../imagenette2-160/val/n03445777/n03445777_15131.JPEG", "fish"),
        ("../imagenette2-160/val/n03445777/n03445777_1482.JPEG", "golf_ball"), # golf_ball
        ("../imagenette2-160/val/n03445777/n03445777_1482.JPEG", "chain_saw"),
        ("../imagenette2-160/val/n03394916/n03394916_15691.JPEG", "french_horn"), # french_horn
        ("../imagenette2-160/val/n03394916/n03394916_15691.JPEG", "fish"),
        ("../imagenette2-160/val/n03028079/n03028079_50060.JPEG", "church"), # church
        ("../imagenette2-160/val/n03028079/n03028079_50060.JPEG", "fish"),
        ("../imagenette2-160/val/n03028079/n03028079_50060.JPEG", "golf_ball"),
        ("../imagenette2-160/val/n01440764/n01440764_13770.JPEG", "fish"), # fish
        ("../imagenette2-160/val/n01440764/n01440764_13770.JPEG", "garbage_truck"),
        ("../imagenette2-160/val/n01440764/n01440764_13770.JPEG", "french_horn"),
        ("../imagenette2-160/val/n01440764/n01440764_13770.JPEG", "chain_saw"),
        ("../imagenette2-160/val/n01440764/n01440764_13770.JPEG", "golf_ball"),
        ("../imagenette2-160/val/n02979186/n02979186_482.JPEG", "casette_player"), # casette_player
        ("../imagenette2-160/val/n02979186/n02979186_482.JPEG", "golf_ball"),
        ("../imagenette2-160/val/n02979186/n02979186_1961.JPEG", "casette_player"), # casette_player
        ("../imagenette2-160/val/n02979186/n02979186_1961.JPEG", "fish"),
        ("../imagenette2-160/val/n03425413/n03425413_16581.JPEG", "gas_pump"), # gas_pump
        ("../imagenette2-160/val/n03425413/n03425413_16581.JPEG", "church")
    ]

    ims_classes = [
        (utils.get_px(fpath)[:, :utils.IM_LEN, :utils.IM_LEN].unsqueeze(dim = 0).to(utils.DEVICE), im_class) # add extra dim for batch size.
        for fpath, im_class in fpaths_classes
    ]

    nets = []
    for fpath in [f"../net_saves/net_{NORMAL_NET_ID}_40.pt", f"../net_saves/net_{SOM_NET_ID}_40.pt"]:
        if utils.MODEL_TYPE == "hw":
            nets.append(design.HwNetworkGlobal(len_output = len(utils.HT_DIR_CLASS)))
        else:
            nets.append(design.VGGUntrained(len_output = len(utils.HT_DIR_CLASS)))

        nets[-1].load_state_dict(torch.load(fpath, weights_only = True, map_location = utils.DEVICE))
        nets[-1].eval()

    som = som_loss.SoftSomLoss2d(
        map_length = SOM_MAP_LENGTH,
        vector_length = nets[0].fc_last_layer.in_features if utils.MODEL_TYPE == "hw" else nets[0].classifier[-1].in_features,
        num_classes = len(utils.HT_DIR_CLASS),
        smoothing_kernel_std = 5
    )
    som.weights = torch.load(f"../net_saves/som_weights_{SOM_NET_ID}_40.pt", weights_only = True, map_location = utils.DEVICE)

    print("Loaded nets, som weights.", flush = True)

    os.mkdir(f"../pics/cam_pic_{SOM_NET_ID}/")

    # explain_cam(ims_classes, nets)
    explain_gradcam(ims_classes, nets, som)
    explain_som_class_representation(som)


if __name__ == "__main__":
    main()
