import pickle

import yaml
import numpy as np
from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.early_stop import no_progress_loss

import random
import torch

weight = np.arange(0., float(1.05e-4), float(5e-5))
params = {}

device = torch.device('cuda')

def w in weight:
    params['ortho_weight'] = w
    res = []

    print('bbbp: {}'.format(params))
    for __ in range(5):
        val_acc = main(**params)
        res.append(-val_acc)

    ret = sum(res) / len(res)
    print("Average Val Acc: {}".format(ret))