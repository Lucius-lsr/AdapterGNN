import argparse
import statistics

from model_lora import GNN_graphpred_lora
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
from tqdm import tqdm

import logging

import warnings
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

warnings.filterwarnings("ignore")

cls_criterion = torch.nn.BCEWithLogitsLoss()
criterion = nn.BCEWithLogitsLoss(reduction="none")


# def train(args, model, device, loader, optimizer):
#     model.train()
#
#     for step, batch in enumerate(loader):
#         batch = batch.to(device)
#         __, pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
#         y = batch.y.view(pred.shape).to(torch.float64)
#
#         # Whether y is non-null or not.
#         is_valid = y ** 2 > 0
#         # Loss matrix
#         loss_mat = criterion(pred.double(), (y + 1) / 2)
#         # loss matrix after removing null target
#         loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
#
#         optimizer.zero_grad()
#         loss = torch.sum(loss_mat) / torch.sum(is_valid)
#         loss.backward()
#
#         optimizer.step()
def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            __, pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                __, pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

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
    # parser.add_argument('--num_layer', type=int, default=5,
    #                     help='number of GNN message passing layers (default: 5).')
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
    # parser.add_argument('--dataset', type=str, default='hiv')  # omit
    # parser.add_argument('--dataset', type=str, default='sider')
    # parser.add_argument('--dataset', type=str, default='tox21')  # {-1.0: 72084, 1.0: 5862, 0.0: 16026}
    # parser.add_argument('--dataset', type=str, default='muv')  # omit {0.0: 1332593, -1.0: 249397, 1.0: 489}
    # parser.add_argument('--dataset', type=str, default='toxcast')  # {-1.0: 1407009, 0.0: 3757732, 1.0: 126651}
    # parser.add_argument('--dataset', type=str, default='pcba')

    # parser.add_argument('--input_model_file', type=str, default='')
    # parser.add_argument('--input_model_file', type=str, default='model_gin/infomax.pth')
    # parser.add_argument('--input_model_file', type=str, default='model_gin/edgepred.pth')
    # parser.add_argument('--input_model_file', type=str, default='model_gin/contextpred.pth')
    parser.add_argument('--input_model_file', type=str, default='model_gin/masking.pth')
    # parser.add_argument('--input_model_file', type=str, default='models_graphcl/graphcl_80.pth')
    # parser.add_argument('--input_model_file', type=str, default='models_simgrace/simgrace_80.pth')

    parser.add_argument('--num_layer', type=int, default=5)

    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=runseed,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--log', type=str)

    # parser.add_argument('--bottleneck_dim', type=int)
    # parser.add_argument('--reserve', type=float)
    parser.add_argument('--middle', type=float, default=2)
    parser.add_argument('--scale', type=float, default=-1)

    args = parser.parse_args()

    logging.basicConfig(format='%(message)s', level=logging.INFO, filename='log/{}.log'.format(args.log), filemode='a')

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
    if args.dataset == 'pcba':
        dataset = PygGraphPropPredDataset('ogbg-molpcba')
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]
        split_idx = dataset.get_idx_split()
        train_dataset, valid_dataset, test_dataset = dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]
    else:
        dataset_root = "../../MoleGraphPrompt/chem/dataset/"
        dataset = MoleculeDataset(dataset_root + args.dataset, dataset=args.dataset)
        if args.split == "scaffold":
            smiles_list = pd.read_csv(dataset_root + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split_multask(args, dataset, smiles_list, null_value=0,
                                                                                frac_train=0.8,
                                                                                frac_valid=0.1, frac_test=0.1)
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
                                                                      frac_test=0.1, seed=args.seed)
        elif args.split == "random_scaffold":
            smiles_list = pd.read_csv(dataset_root + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0,
                                                                               frac_train=0.8, frac_valid=0.1,
                                                                               frac_test=0.1, seed=args.seed)
        else:
            raise ValueError("Invalid split option.")


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # set up model
    model = GNN_graphpred_gp(args, args.num_layer, args.emb_dim, num_tasks, JK=args.JK,
                                 drop_ratio=args.dropout_ratio,
                                 graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    model.to(device)

    # baseline
    if type(model) is GNN_graphpred:
        model_param_group = []
        model_param_group.append({"params": model.gnn.parameters()})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr})

    # gp_311
    if type(model) is GNN_graphpred_gp:
        model_param_group = []
        model_param_group.append({"params": model.gnn.prompts.parameters(), "lr": args.lr})
        model_param_group.append({"params": model.gnn.gating_parameter, "lr": args.lr})
        for name, p in model.gnn.named_parameters():
            if name.startswith('batch_norms'):
                model_param_group.append({"params": p})
            if 'mlp' in name and name.endswith('bias'):
                model_param_group.append({"params": p})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr})

    # lora
    if type(model) is GNN_graphpred_lora:
        model_param_group = []
        for name, p in model.gnn.gnns.named_parameters():
            if 'lora' in name or 'scale' in name or 'ia3' in name:
                model_param_group.append({"params": p})
        for name, p in model.gnn.named_parameters():
            if name.startswith('batch_norms'):
                model_param_group.append({"params": p})
            if 'mlp' in name and name.endswith('bias'):
                model_param_group.append({"params": p})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr})

    # # add
    # if type(model) is GNN_graphpred_add:
    #     model_param_group = []
    #     model_param_group.append({"params": model.gnn.feature_add, "lr": args.lr})
    #     for name, p in model.gnn.named_parameters():
    #         if name.startswith('batch_norms'):
    #             model_param_group.append({"params": p})
    #     model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr})
    #
    # # prompt
    # if type(model) is GNN_graphpred_prompt:
    #     model_param_group = []
    #     model_param_group.append({"params": model.gnn.virtual, "lr": args.lr})
    #     model_param_group.append({"params": model.gnn.virtual_edge, "lr": args.lr})
    #     for name, p in model.gnn.named_parameters():
    #         if name.startswith('batch_norms'):
    #             model_param_group.append({"params": p})
    #     model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    assoc_train_acc = -1
    best_val_acc = -1
    assoc_test_acc = -1

    evaluator = Evaluator('ogbg-molpcba')

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer)
        if args.eval_train:
            train_acc = eval(model, device, train_loader, evaluator)['ap']
        else:
            train_acc = 0
        val_acc = eval(model, device, val_loader, evaluator)['ap']
        test_acc = eval(model, device, test_loader, evaluator)['ap']

        print({'Train': train_acc, 'Validation': val_acc, 'Test': test_acc})

        if val_acc > best_val_acc:
            assoc_train_acc = train_acc
            best_val_acc = val_acc
            assoc_test_acc = test_acc

    return assoc_test_acc


if __name__ == "__main__":
    overall = []
    quick_exp = ['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast']
    full_exp = ['bace', 'bbbp', 'clintox', 'hiv', 'sider', 'tox21', 'muv', 'toxcast']

    for dataset in ['pcba']:
        total_acc = []
        repeat = 1
        for runseed in tqdm(range(repeat)):
            acc = main(runseed, dataset)
            total_acc.append(acc)
            overall.append(acc)
        logging.info('{:.2f}±{:.2f}'.format(100 * sum(total_acc) / len(total_acc),
                                            100 * statistics.pstdev(total_acc)))
    logging.info('----------')
