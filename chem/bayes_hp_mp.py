from finetune_motif import main

import pickle

import yaml
import numpy as np
from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.early_stop import no_progress_loss

import random
import torch

dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
n_heads = [2, 4, 8]

device = torch.device('cuda')
params = {}

for d in dropout:
    for n in n_heads:
        params['lr'] = 0.001
        params['threshold'] = 10
        params['enc_dropout'] = d
        params['tfm_dropout'] = d
        params['dec_dropout'] = d
        params['enc_ln'] = False
        params['tfm_ln'] = True
        params['conc_ln'] = False
        
        params['vocab'] = 'mgssl'
        params['init'] = 'zeros'
        params['num_heads'] = int(n)

        res = []
        print('sider: {}'.format(params))

        for __ in range(1):
            val_acc = main(**params)
            res.append(-val_acc)

        ret = sum(res) / len(res)
        print(ret)
