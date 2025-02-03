import torchvision
import typing
import torch
# import PIL

IM_LEN = 160
EPOCH_CNT = 30
DEBUG_SAVE_EVERY = 5

HT_DIR_CLASS = {
    "n01440764": "fish",
    "n02102040": "dog",
    "n02979186": "casette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute"
}

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# presupune ca imaginea primita este patrat, kernel size-ul e patrat, padding, stride, dim nu au valori diferite pe axe.
def compute_conv2d_out_size(in_size: int, conv: typing.Union[torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.ReLU]) -> int:
    if type(conv) == torch.nn.Conv2d:
        return (in_size + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1

    elif type(conv) == torch.nn.MaxPool2d:
        return (in_size + 2 * conv.padding - conv.dilation * (conv.kernel_size - 1) - 1) // conv.stride + 1

    elif type(conv) == torch.nn.AvgPool2d:
        return (in_size + 2 * conv.padding - conv.kernel_size) // conv.stride + 1

    return in_size

def get_px(fpath: str):
    # im = PIL.Image.open(fpath)
    # return torch.FloatTensor([[im.getdata()[y * im.size[0] + x] for x in range(im.size[0])] for y in range(im.size[1])]).permute(2, 0, 1) / 255
    
    im = torchvision.io.decode_image(fpath) / 255
    if im.shape[0] == 1: #posibil ca imaginea primita sa fie uni-canal.
        im = im.broadcast_to([3, *im.shape[1:]])

    return im

def debug_ht_float_content(ht: dict):
    return ", ".join([f"{key}: {round(value, 3)}" for key, value in ht.items()])
