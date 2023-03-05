import argparse
import statistics

from model_gp import GNN_graphpred_gp
from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, scaffold_split_multask
import pandas as pd

import os
import shutil

import warnings

warnings.filterwarnings("ignore")

criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        __, pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            __, pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    # if len(roc_list) < y_true.shape[1]:
    #     print("Some target is missing!")
    #     print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list)  # y_true.shape[1]


def main(runseed, dataset):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")

    parser.add_argument('--dataset', type=str, default=dataset)

    # parser.add_argument('--dataset', type=str, default='bace')
    # parser.add_argument('--dataset', type=str, default='bbbp')
    # parser.add_argument('--dataset', type=str, default='clintox')
    # parser.add_argument('--dataset', type=str, default='hiv')
    # parser.add_argument('--dataset', type=str, default='sider')
    # parser.add_argument('--dataset', type=str, default='tox21')  # {-1.0: 72084, 1.0: 5862, 0.0: 16026}
    # parser.add_argument('--dataset', type=str, default='muv')  # {0.0: 1332593, -1.0: 249397, 1.0: 489}
    # parser.add_argument('--dataset', type=str, default='toxcast')  # {-1.0: 1407009, 0.0: 3757732, 1.0: 126651}

    # parser.add_argument('--input_model_file', type=str, default='model_gin/masking.pth')
    # parser.add_argument('--input_model_file', type=str, default='models_graphcl/graphcl_80.pth')
    # parser.add_argument('--input_model_file', type=str, default='')
    # parser.add_argument('--input_model_file', type=str, default='model_gin/infomax.pth')
    # parser.add_argument('--input_model_file', type=str, default='model_gin/edgepred.pth')
    parser.add_argument('--input_model_file', type=str, default='model_gin/contextpred.pth')

    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=runseed,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    # set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split_multask(dataset, smiles_list, null_value=0,
                                                                            frac_train=0.8,
                                                                            frac_valid=0.1, frac_test=0.1)
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
                                                                  frac_test=0.1, seed=args.seed)
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0,
                                                                           frac_train=0.8, frac_valid=0.1,
                                                                           frac_test=0.1, seed=args.seed)
    else:
        raise ValueError("Invalid split option.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    model.to(device)

    # baseline
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})

    # gp
    # model_param_group = []
    # model_param_group.append({"params": model.gnn.prompt.parameters(), "lr": args.lr})
    # # model_param_group.append({"params": model.gnn.gating_parameter, "lr": args.lr})
    # for name, p in model.gnn.named_parameters():
    #     if name.startswith('batch_norms'):
    #         model_param_group.append({"params": p})
    #     elif not name.startswith('prompt') and name.endswith('bias'):
    #         model_param_group.append({"params": p})
    # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    assoc_train_acc = -1
    best_val_acc = -1
    assoc_test_acc = -1

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer)
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        # print("train: %f val: %f test: %f" % (train_acc, val_acc, test_acc))

        if val_acc > best_val_acc:
            assoc_train_acc = train_acc
            best_val_acc = val_acc
            assoc_test_acc = test_acc

    return assoc_test_acc


if __name__ == "__main__":
    overall = []
    for dataset in ['bace', 'bbbp', 'clintox', 'hiv', 'sider', 'tox21', 'muv', 'toxcast']:
        total_acc = []
        repeat = 10
        for runseed in range(repeat):
            acc = main(runseed, dataset)
            total_acc.append(acc)
            overall.append(acc)
        print('Average acc:{:.2f}±{:.2f}'.format(100 * sum(total_acc) / len(total_acc),
                                                 100 * statistics.pstdev(total_acc)))
    print()
    print(100 * sum(overall) / len(overall))