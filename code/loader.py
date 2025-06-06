import torch.utils.data
import torchvision
import random
import torch
import os

import utils

class Dataset(torch.utils.data.Dataset):
    #Initialization.
    def __init__(self, dset_type: str, class_ht = None, break_after_fill_ht = False):
        assert(dset_type in ["train", "test"])

        self.dset = [] # [(X, y)].

        self.class_ht = class_ht if class_ht is not None else {} #trebuie la build test sa tin minte aceeasi bijectie intre clase si id.
        for root, dirs, files in os.walk(f"../imagenette2-160/{'train' if dset_type == 'train' else 'val'}"):
            for fname in files:
                #e.g. file at ../imagenette2-160/train/n02102040/ILSVRC2012_val_00043994.JPEG
                #e.g. file at ../imagenette2-160/train/n03888257/n03888257_12775.JPEG
                fpath = os.path.join(root, fname)
                if fpath.endswith(".JPEG"):
                    class_str = fpath.split('/')[-2]
                    if class_str not in self.class_ht:
                        self.class_ht[class_str] = len(self.class_ht)

                    self.dset.append((
                        utils.get_px(fpath),
                        self.class_ht[class_str]
                    ))

                if break_after_fill_ht and len(self.class_ht) >= len(utils.HT_DIR_CLASS):
                    break

        self.vgg_resizer = torchvision.transforms.Resize((240, 240), antialias = None)

        # TODO vezi cum poti sa pui tot self.dset direct pe gpu.
        print(f"Finished loading Dataset ({dset_type = }) {'(! broke after filling ht)' if break_after_fill_ht else ''}.")

    #Denotes the total number of samples.
    def __len__(self):
        return len(self.dset)

    #Generates one sample of data.
    def __getitem__(self, ind):
        h, w = self.dset[ind][0].shape[1:]
        # minimul dintre cele doua dimensiuni este IM_LEN. iau aleator o bucata IM_LEN x IM_LEN din imagine.

        y, x = random.randint(0, h - utils.IM_LEN), random.randint(0, w - utils.IM_LEN)
        im = self.dset[ind][0][:, y: y+utils.IM_LEN, x: x+utils.IM_LEN]

        if random.randint(0, 1): #augmentare prin flip pe verticala.
            im = im.flip(dims = [-1])

        if utils.MODEL_TYPE == "vgg":
            im = self.vgg_resizer(im)

        return im, self.dset[ind][1]
