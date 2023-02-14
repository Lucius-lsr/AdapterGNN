from finetune_mask import main

import pickle

import yaml
import numpy as np
from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.early_stop import no_progress_loss

import random
import torch

hp_space = {'ortho_weight': hp.quniform('ortho_weight', 0., 1e-4, 2.5e-6),
            'num_clusters': hp.quniform('num_clusters', 1, 100, 2)}

device = torch.device('cuda')

def objective(params):
    params['ortho_weight'] = float(params['ortho_weight'])
    params['num_clusters'] = int(params['num_clusters'])

    res = []

    print('tox21: {}'.format(params))

    for __ in range(1):
        val_acc = main(**params)
        res.append(-val_acc)

    ret = sum(res) / len(res)
    print('Average Val Acc: {}'.format(ret))
    return ret

trials = Trials()

best = fmin(objective, hp_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=1000)

print(best)
space_eval(hp_space, best)