import torchvision
import torch

import som_loss
import utils

class HwNetworkGlobal(torch.nn.Module):
    def __init__(self, len_output: int):
        super(HwNetworkGlobal, self).__init__()

        #conv2d -> max pool -> ReLU -> conv2d -> ReLU -> concat -> FC. (+ global pooling)

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 3, out_channels = 30, kernel_size = 11, device = utils.DEVICE),
            torch.nn.AvgPool2d(kernel_size = 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 5, device = utils.DEVICE),
            torch.nn.AvgPool2d(kernel_size = 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 30, out_channels = 50, kernel_size = 3, device = utils.DEVICE),
            torch.nn.ReLU()
        )

        im_len = utils.IM_LEN
        for layer in self.conv_layers:
            im_len = utils.compute_conv2d_out_size(im_len, layer)
        fc_insize = im_len ** 2 * self.conv_layers[-2].out_channels

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(fc_insize, 128, device = utils.DEVICE),
            torch.nn.BatchNorm1d(128, device = utils.DEVICE),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        self.fc_last_layer = torch.nn.Linear(128 + self.conv_layers[-2].out_channels, len_output, device = utils.DEVICE)


    def forward(self, x):
        out_conv = self.conv_layers(x)

        out_sll = self.fc_layers(out_conv.view(x.shape[0], -1))
        out_sll = torch.hstack([out_sll, out_conv.sum(dim = [-2, -1])])
        
        out = self.fc_last_layer(out_sll)

        return out, out_sll, out_conv

class VGGUntrained(torch.nn.Module):
    def __init__(self, len_output: int):
        super(VGGUntrained, self).__init__()
        og_vgg = torchvision.models.vgg11_bn()

        self.features = og_vgg.features.to(utils.DEVICE)
        self.avgpool = og_vgg.avgpool.to(utils.DEVICE) # TODO: VGG doesn't have GAP.
        self.classifier = og_vgg.classifier.to(utils.DEVICE)

        self.classifier[0] = torch.nn.Linear(25088, 768, device = utils.DEVICE)
        self.classifier[3] = torch.nn.Linear(768, 768, device = utils.DEVICE)
        self.classifier[6] = torch.nn.Linear(768, len_output, device = utils.DEVICE)

        del og_vgg

    def forward(self, x):
        out_conv = self.avgpool(self.features(x))
        out_sll = self.classifier[:-3](out_conv.view(x.shape[0], -1))
        out = self.classifier[-3:](out_sll) # the last three layers are ReLU, Dropout and FC.

        return out, out_sll, out_conv
