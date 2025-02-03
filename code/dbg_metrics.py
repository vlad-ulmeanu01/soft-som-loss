from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import time
import sys

import main
import design
import loader
import utils

def dbg_main():
    net = design.HwNetworkGlobal(len_output = len(utils.HT_DIR_CLASS))
    net.load_state_dict(torch.load("../net_saves/net_1738586107_40.pt", weights_only = True))
    net.eval()

    criterion = torch.nn.CrossEntropyLoss()
    
    print("1", flush = True)

    dsets = {}
    dsets["train"] = loader.Dataset(dset_type = "train")
    dsets["test"] = loader.Dataset(dset_type = "test", class_ht = dsets["train"].class_ht)

    print("2", flush = True)

    gens = {}
    gens["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size = len(dsets["test"]))
    
    print("3", flush = True)

    ht_index_to_class = {dsets["train"].class_ht[dirname]: utils.HT_DIR_CLASS[dirname] for dirname in dsets["train"].class_ht}
    index_to_class = [x for _, x in sorted(ht_index_to_class.items())]
    
    print(f"class_ht = {dsets['train'].class_ht}.")
    print(f"{ht_index_to_class = }")
    print(f"{index_to_class = }")

    metrics = {dset_type: {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "confusion": [], "report": []} for dset_type in dsets}

    yt, yp = main.run_net(net, None, criterion, None, gens, metrics, "test")

    accuracy = accuracy_score(yt, yp)
    precision, recall, f1, _ = precision_recall_fscore_support(yt, yp, average = "macro", zero_division = 0.0)
    metrics["test"]["precision"].append(precision)
    metrics["test"]["recall"].append(recall)
    metrics["test"]["f1"].append(f1)
    cm = confusion_matrix(yt, yp, normalize = "true")
    report = classification_report(yt, yp, target_names = index_to_class, zero_division = 0.0)

    print(f"{accuracy = }")
    print(f"{precision = }")
    print(f"{recall = }")
    print(f"{f1 = }")
    print(f"{cm = }")
    print(f"{report = }")

    pass

if __name__ == "__main__":
    dbg_main()
