import matplotlib.pyplot as plt
import itertools
import torch

import design
import loader
import utils


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
    for fpath in ["../net_saves/net_1731409474_100.pt", "../net_saves/net_1738586107_40.pt"]:
        nets.append(design.HwNetworkGlobal(len_output = len(utils.HT_DIR_CLASS)))
        nets[-1].load_state_dict(torch.load(fpath, weights_only = True))
        nets[-1].eval()

    print("allclose conv_layers[0]: ", torch.allclose(nets[0].conv_layers[0].weight, nets[1].conv_layers[0].weight))
    print("allclose conv_layers[3]: ", torch.allclose(nets[0].conv_layers[3].weight, nets[1].conv_layers[3].weight))
    print("allclose conv_layers[6]: ", torch.allclose(nets[0].conv_layers[6].weight, nets[1].conv_layers[6].weight))
    print("allclose fc_layers[0]: ", torch.allclose(nets[0].fc_layers[0].weight, nets[1].fc_layers[0].weight))
    print("allclose last layer: ", torch.allclose(nets[0].fc_last_layer.weight, nets[1].fc_last_layer.weight))

    print("Loaded nets.", flush = True)

    # class_ht = {'n02102040': 0, 'n03445777': 1, 'n03394916': 2, 'n03425413': 3, 'n03000684': 4, 'n01440764': 5, 'n03888257': 6, 'n03417042': 7, 'n02979186': 8, 'n03028079': 9}.
    # ht_index_to_class = {0: 'dog', 1: 'golf_ball', 2: 'french_horn', 3: 'gas_pump', 4: 'chain_saw', 5: 'fish', 6: 'parachute', 7: 'garbage_truck', 8: 'casette_player', 9: 'church'}
    # index_to_class = ['dog', 'golf_ball', 'french_horn', 'gas_pump', 'chain_saw', 'fish', 'parachute', 'garbage_truck', 'casette_player', 'church']

    # class_names = list(utils.HT_DIR_CLASS.values())
    class_names = ['dog', 'golf_ball', 'french_horn', 'gas_pump', 'chain_saw', 'fish', 'parachute', 'garbage_truck', 'casette_player', 'church']

    num_channels = nets[0].conv_layers[-2].out_channels
    for i, (im, im_class) in zip(itertools.count(), ims_classes):
        fig, ax = plt.subplots(1, 3, figsize = (15, 15))

        wanted_class = class_names.index(im_class) # index of wanted class, here in [0, 9].

        ax[0].imshow(torch.permute(im[0].cpu(), (1, 2, 0)).detach().numpy())
        ax[0].set_title("Original image")

        for net, msg, j in zip(nets, ["normal net", "net with SOM loss"], range(1, 3)):
            # im.shape: [1, 3, 160, 160]
            out, out_sll, out_conv = net(im) # out_conv: torch.Size([1, 50, 33, 33]) -- 50 feature map outputs.

            sm = torch.nn.functional.softmax(out.detach()[0], dim = 0)
            pred_class = class_names[sm.argmax().item()]

            print(f"{j = }, out softmax = {torch.round(sm, decimals = 3)}, {pred_class = }", flush = True)

            # torch.Size([50, 1, 1]):
            kernel_weights = net.fc_last_layer.weight[wanted_class, net.fc_last_layer.in_features - num_channels:].unsqueeze(dim = -1).unsqueeze(dim = -1)

            cam = torch.nn.functional.relu((out_conv * kernel_weights)[0]).mean(dim = 0)

            ax[j].imshow(cam.cpu().detach().numpy())
            ax[j].set_title(f"({msg}, predicted {pred_class} with\nproba {round(sm.max().item(), 2)}), CAM for {im_class} (proba {round(sm[wanted_class].item(), 2)})")

        fig.savefig(f"../pics/cam_pic_{i}.png", bbox_inches = "tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
