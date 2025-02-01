from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import time
import json
import sys

import som_loss
import design
import loader
import utils


def run_net(net, soft_som_loss, criterion, optimizer, gens, metrics, dset_type: str) -> list: # intoarce [(yTruth, yPred)].
    with torch.set_grad_enabled(dset_type == "train"), torch.autograd.set_detect_anomaly(True):
        metrics[dset_type]["loss"].append(0.0)

        pair_count, all_yTruth, all_yPred = 0, [], []
        for x, yTruth_indexes in gens[dset_type]:
            x, yTruth_indexes = x.to(utils.DEVICE), yTruth_indexes.to(utils.DEVICE)

            yPred, yPred_sll, _ = net(x)
            yTruth = torch.nn.functional.one_hot(yTruth_indexes).float()

            if dset_type == "train":
                optimizer.zero_grad()

                loss = criterion(yPred, yTruth) + soft_som_loss(yPred_sll, yTruth_indexes)
                # loss = criterion(yPred, yTruth)
                # loss.backward(retain_graph = True)
                # loss = soft_som_loss(yPred_sll, yTruth_indexes)
                loss.backward()

                optimizer.step()
            else:
                loss = criterion(yPred, yTruth)

            metrics[dset_type]["loss"][-1] += loss.detach().item() * x.shape[0]
            pair_count += x.shape[0]
            print(f"ok {pair_count = }.", flush = True)

            yPred_indexes = yPred.argmax(dim = 1)
            all_yTruth.append(yTruth_indexes.to("cpu"))
            all_yPred.append(yPred_indexes.to("cpu"))

        print(f"Ended generator for loop ({dset_type = }).", flush = True)

        metrics[dset_type]["loss"][-1] = metrics[dset_type]["loss"][-1] / pair_count if pair_count else metrics[dset_type]["loss"][-1]
        return torch.cat(all_yTruth), torch.cat(all_yPred)


def main():
    t_start = time.time()
    runid = str(int(t_start))

    net = design.HwNetworkGlobal(len_output = len(utils.HT_DIR_CLASS))
    soft_som_loss = som_loss.SoftSomLoss2d(map_length = 20, vector_length = net.fc_last_layer.in_features, num_classes = len(utils.HT_DIR_CLASS), lr = 1e-3, smoothing_kernel_std = 1)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters()) # + [soft_som_loss.weights]

    print(f"({runid = }, {net = }, {criterion = }, {optimizer.param_groups = }, working on device: {utils.DEVICE}.")
    sys.stdout.flush()

    dsets = {}
    dsets["train"] = loader.Dataset(dset_type = "train")
    dsets["test"] = loader.Dataset(dset_type = "test", class_ht = dsets["train"].class_ht)

    gens = {}
    gens["train"] = torch.utils.data.DataLoader(dsets["train"], batch_size = 128, shuffle = True)
    gens["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size = len(dsets["test"]))

    ht_index_to_class = {dsets["train"].class_ht[dirname]: utils.HT_DIR_CLASS[dirname] for dirname in dsets["train"].class_ht}
    index_to_class = [x for _, x in sorted(ht_index_to_class.items())]

    print(f"init time taken: {round(time.time() - t_start, 3)}s."); t_start = time.time()
    sys.stdout.flush()

    metrics = {dset_type: {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "confusion": [], "report": []} for dset_type in dsets}
    for epoch in range(1, utils.EPOCH_CNT + 1):
        for dset_type in ["train", "test"]:
            yt, yp = run_net(net, soft_som_loss, criterion, optimizer, gens, metrics, dset_type)

            metrics[dset_type]["accuracy"].append(accuracy_score(yt, yp))
            precision, recall, f1, _ = precision_recall_fscore_support(yt, yp, average = "macro", zero_division = 0.0)
            metrics[dset_type]["precision"].append(precision)
            metrics[dset_type]["recall"].append(recall)
            metrics[dset_type]["f1"].append(f1)
            metrics[dset_type]["confusion"].append(confusion_matrix(yt, yp, normalize = "true"))
            metrics[dset_type]["report"].append(classification_report(yt, yp, target_names = index_to_class, zero_division = 0.0))

        if epoch % utils.DEBUG_SAVE_EVERY == 0 or epoch == utils.EPOCH_CNT:
            torch.save(net.state_dict(), f"../net_saves/net_{runid}_{epoch}.pt")
            torch.save(soft_som_loss.weights, f"../net_saves/som_weights_{runid}_{epoch}.pt")

            print(f"{epoch = }: ")
            print(f"train report: {metrics['train']['report'][-1]}")
            print(f"test report: {metrics['test']['report'][-1]}")
            sys.stdout.flush()

        print(f"{epoch = } took {round(time.time() - t_start, 3)}s."); t_start = time.time()
        sys.stdout.flush()

    fig, ax = plt.subplots(2, 3, figsize = (25, 17))
    
    ax[0, 0].set_ylim((0, 5)) # loss.
    ax[0, 1].set_ylim((0, 1)); ax[0, 2].set_ylim((0, 1)); ax[1, 0].set_ylim((0, 1)); ax[1, 1].set_ylim((0, 1)) # accuracy, precision, recall, f1.

    for metric, loc in zip(["loss", "accuracy", "precision", "recall", "f1"], [(i, j) for i in range(ax.shape[0]) for j in range(ax.shape[1])]):
        ax[loc[0], loc[1]].plot(metrics["train"][metric])
        ax[loc[0], loc[1]].plot(metrics["test"][metric])
        ax[loc[0], loc[1]].legend(["train", "test"])
        ax[loc[0], loc[1]].set_title(metric)

    disp = ConfusionMatrixDisplay(metrics["test"]["confusion"][-1], display_labels = index_to_class)
    disp.plot(ax = ax[-1, -1], xticks_rotation = 45)
    disp.ax_.set_title("(Test) Confusion matrix post training")

    fig.savefig(f"../net_logs/net_{runid}_losses.png", bbox_inches = "tight")

    for dset_type in ["train", "test"]:
        del metrics[dset_type]["confusion"]
        del metrics[dset_type]["report"]

        with open(f"../net_logs/metrics_log_{runid}_{dset_type}.json", "w") as fout:
            json.dump(metrics[dset_type], fout, indent = 4)

        print(f"{dset_type = }, {metrics[dset_type] = }")


if __name__ == "__main__":
    main()

