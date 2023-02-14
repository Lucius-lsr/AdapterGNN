import argparse
import statistics

from model_p import GNN_prompt_graphpred
from loader import MoleculeDataset
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# criterion = nn.BCEWithLogitsLoss(reduction = "none")
criterion = nn.CrossEntropyLoss()


def train(args, target, model, device, loader, optimizer):
    model.train()

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        __, pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # y = batch.y.view(pred.shape).to(torch.float64)
        batch.y = batch.y.view(pred.shape[0], -1)
        if len(batch.y.shape) > 1:
            y = batch.y[:, target].flatten().to(torch.long)
        else:
            y = batch.y[target].flatten().to(torch.long)

        # #Whether y is non-null or not.
        # is_valid = y**2 > 0
        # #Loss matrix
        # loss_mat = criterion(pred.double(), (y+1)/2)
        # #loss matrix after removing null target
        # loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        # loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss = criterion(pred, y)
        if model.gnn.method == 'qkv':
            loss += model.gnn.sim_loss
        loss.backward()

        optimizer.step()


def eval(args, target, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            __, pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        pred = F.softmax(pred, dim=-1)

        # y_true.append(batch.y.view(pred.shape))
        # y_scores.append(pred)
        batch.y = batch.y.view(pred.shape[0], -1)
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

    # y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    # y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    # roc_list = []
    # for i in range(y_true.shape[1]):
    #     #AUC is only defined when there is at least one positive data.
    #     if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
    #         is_valid = y_true[:,i]**2 > 0
    #         roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    # #if len(roc_list) < y_true.shape[1]:
    # #    print("Some target is missing!")
    # #    print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    # return sum(roc_list)/len(roc_list) #y_true.shape[1]

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    return roc_auc_score(y_true, y_scores[:, 1])


def main(runseed):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=2,
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
    parser.add_argument('--dataset', type=str, default='bbbp',
                        help='root directory of dataset. For now, only classification.')
    # parser.add_argument('--input_model_file', type=str, default='model_gin/masking.pth',
    #                     help='filename to read the model (if there is any)')
    parser.add_argument('--input_model_file', type=str, default='models_graphcl/graphcl_80.pth',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=runseed,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=1, help='evaluating training or not')
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
    assoc_test_acc_list = []
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
            continue

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # set up model
        model = GNN_prompt_graphpred(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                                     graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        if not args.input_model_file == "":
            model.from_pretrained(args.input_model_file)

        model.to(device)

        model.gnn.init_node_prompt(train_loader)
        # model.gnn.load_prompt('masking_prompt_atte_mul_0206.pth', args.input_model_file)

        # set up optimizer
        # different learning rate for different part of GNN
        model_param_group = []
        # model_param_group.append({"params": model.gnn.parameters()})
        model_param_group.append({"params": model.gnn.virtual})
        model_param_group.append({"params": model.gnn.virtual_edge})
        model_param_group.append({"params": model.gnn.node_mlp.parameters()})
        model_param_group.append({"params": model.gnn.edge_mlp.parameters()})
        model_param_group.append({"params": model.gnn.sequential_prompt.parameters()})

        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})

        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        # print(optimizer)

        train_acc_list = []
        val_acc_list = []
        test_acc_list = []

        assoc_train_acc = -1
        best_val_acc = -1
        assoc_test_acc = -1

        for epoch in range(1, args.epochs + 1):
            # print("====epoch " + str(epoch))

            train(args, target, model, device, train_loader, optimizer)

            # print("====Evaluation")
            if args.eval_train:
                train_acc = eval(args, target, model, device, train_loader)
            else:
                #    print("omit the training accuracy computation")
                train_acc = 0
            val_acc = eval(args, target, model, device, val_loader)
            test_acc = eval(args, target, model, device, test_loader)

            print("train: %f val: %f test: %f" % (train_acc, val_acc, test_acc))

            if val_acc > best_val_acc:
                assoc_train_acc = train_acc
                best_val_acc = val_acc
                assoc_test_acc = test_acc

            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            train_acc_list.append(train_acc)

            # print("")

        print("assoc train: %f best val: %f assoc test: %f" % (assoc_train_acc, best_val_acc, assoc_test_acc))
        assoc_test_acc_list.append(assoc_test_acc)
    return assoc_test_acc_list


if __name__ == "__main__":
    total_acc = []
    for runseed in range(10):
        accs = main(runseed)
        total_acc += accs
    print('Average acc:{:.2f}±{:.2f}'.format(100 * sum(total_acc) / len(total_acc), 100 * statistics.pstdev(total_acc)))
