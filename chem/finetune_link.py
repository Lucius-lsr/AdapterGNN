import argparse

from loader import MoleculeDataset
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model_link import GNN, GNN_link
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def _ortho_constraint(device, prompt):
    return torch.norm(torch.mm(prompt, prompt.T) - torch.eye(prompt.shape[0]).to(device))


def train(args, kwargs, target, model, device, loader, optimizer):
    model.train()

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        optimizer.zero_grad()

        if len(batch.y.shape) > 1:
            y = batch.y[:, target].flatten().to(torch.long)
        else:
            y = batch.y[target].flatten().to(torch.long)

        preds, embs = model.forward_cl(batch.x, y, batch.edge_index, batch.edge_attr, batch.batch, device)
        try:
            loss = model.loss_cl(preds, embs)
        except IndexError:
            preds = preds.unsqueeze(0)
            embs = embs.unsqueeze(0)
            loss = model.loss_cl(preds, embs)
        loss += kwargs['ortho_weight'] * _ortho_constraint(device, model.get_label_emb())

        loss.backward()
        optimizer.step()


def eval(args, kwargs, target, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            preds, __ = model.forward_cl(batch.x, batch.y, batch.edge_index, batch.edge_attr, batch.batch, device)

        preds = F.normalize(preds, dim=1)
        embs = model.get_label_emb()
        embs = F.normalize(embs, dim=1)
        emb_0 = torch.index_select(embs, 0, torch.tensor([i for i in range(kwargs['num_clusters'])]).to(device))
        emb_0 = emb_0.mean(dim=0, keepdim=True)
        emb_0 = F.normalize(emb_0, dim=1)
        emb_1 = torch.index_select(embs, 0,
                                   torch.tensor([kwargs['num_clusters'] + i for i in range(kwargs['num_clusters'])]).to(
                                       device))
        emb_1 = emb_1.mean(dim=0, keepdim=True)
        emb_1 = F.normalize(emb_1, dim=1)
        embs = torch.cat((emb_0, emb_1), dim=0)

        similarities = torch.mm(preds, embs.T)
        pred = F.softmax(similarities, dim=-1)

        if len(batch.y.shape) > 1:
            y = batch.y[:, target].flatten()
        else:
            y = batch.y[target].flatten()

        if device == 'cpu':
            y_true.extend(y.numpy())
            y_scores.extend(pred.detach().numpy())
        else:
            y_true.extend(y.cpu().numpy())
            y_scores.extend(pred.cpu().detach().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    return roc_auc_score(y_true, y_scores[:, 1])


def main(**kwargs):
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
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='tox21',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_model_file', type=str, default='new_models_graphcl/graphcl.pth',
                        help='filename to read the gnn model (if there is any)')
    parser.add_argument('--proj_head_file', type=str, default='new_models_graphcl/graphcl_head.pth',
                        help='filename to read the projection head weights')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    args = parser.parse_args()

    print(args, kwargs)

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
    elif args.dataset == "hiv":
        num_tasks = 1
        target_list = ["HIV_active"]
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]
    elif args.dataset == "bace":
        num_tasks = 1
        target_list = ["Class"]
    elif args.dataset == "bbbp":
        num_tasks = 1
        target_list = ["p_np"]
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions", "Endocrine disorders",
            "Surgical and medical procedures", "Vascular disorders",
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders", "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders", "Cardiac disorders",
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    elif args.dataset == "clintox":
        num_tasks = 2
        target_list = ['CT_TOX', 'FDA_APPROVED']
    else:
        raise ValueError("Invalid dataset name.")

    # set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    # print(dataset)

    total_val_acc = []
    for target in range(len(target_list)):
        if args.split == "scaffold":
            smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, task_idx=target,
                                                                        null_value=np.nan, frac_train=0.8,
                                                                        frac_valid=0.1, frac_test=0.1)
            # print("scaffold")
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(dataset, task_idx=target, null_value=np.nan,
                                                                      frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                                                                      seed=args.seed)
            # print("random")
        elif args.split == "random_scaffold":
            smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, task_idx=target,
                                                                               null_value=np.nan, frac_train=0.8,
                                                                               frac_valid=0.1, frac_test=0.1,
                                                                               seed=args.seed)
            # print("random scaffold")
        else:
            raise ValueError("Invalid split option.")

        if train_dataset is None:
            print("Task omitted!")
            total_val_acc.append(1.0)
            continue

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # set up model
        gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
        model = GNN_link(kwargs['num_clusters'], gnn)
        if not (args.gnn_model_file == "" or args.proj_head_file == ""):
            model.from_pretrained(args.gnn_model_file, args.proj_head_file)
        else:
            raise ValueError("No pretrained file provided for GNN or projection head.")

        model.to(device)

        with torch.no_grad():
            label_feats = []
            labels = []

            for batch in train_loader:
                batch = batch.to(device)
                pred, __ = model.forward_cl(batch.x, batch.y, batch.edge_index, batch.edge_attr, batch.batch, device)
                label_feats.append(pred)
                labels.append(batch.y)

            label_feats = torch.cat(label_feats)
            labels = torch.cat(labels)

            linit0 = torch.mean(label_feats[torch.nonzero(labels == 0)[:, 0]], dim=0)
            linit1 = torch.mean(label_feats[torch.nonzero(labels == 1)[:, 0]], dim=0)

            label_feats = torch.vstack((linit0, linit1)).to(device)

            cluster_indices = [0 for i in range(kwargs['num_clusters'])]
            cluster_indices.extend([1 for i in range(kwargs['num_clusters'])])

            label_feats = torch.index_select(label_feats, 0, torch.tensor(cluster_indices).to(device))

        if not (args.gnn_model_file == "" or args.proj_head_file == ""):
            model.from_pretrained(args.gnn_model_file, args.proj_head_file)
            model.init_label_emb(label_feats)

        model.to(device)

        # set up optimizer
        # different learning rate for different part of GNN
        # model_param_group = []
        # model_param_group.append({"params": model.gnn.parameters()})
        # if args.graph_pooling == "attention":
        #     model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
        # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})

        # optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        # print(optimizer)

        best_val_acc = -1
        ass_test_acc = -1
        avg_val_acc = []

        for epoch in range(1, args.epochs + 1):
            # print("====epoch " + str(epoch))

            train(args, kwargs, target, model, device, train_loader, optimizer)

            # print("====Evaluation")
            if args.eval_train:
                train_acc = eval(args, kwargs, target, model, device, train_loader)
            else:
                #    print("omit the training accuracy computation")
                train_acc = 0
            val_acc = eval(args, kwargs, target, model, device, val_loader)
            test_acc = eval(args, kwargs, target, model, device, test_loader)

            # print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

            avg_val_acc.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ass_test_acc = test_acc

            # print("")

        avg_val_acc = sum(avg_val_acc) / len(avg_val_acc)
        total_val_acc.append(avg_val_acc)
        print(ass_test_acc)
        # print("val: %f, test: %f" %(avg_val_acc, ass_test_acc))
    return sum(total_val_acc) / len(total_val_acc)


if __name__ == "__main__":
    for _ in range(10):
        main(num_clusters=60, ortho_weight=7.5e-5)
