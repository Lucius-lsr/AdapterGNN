#!/bin/bash

device=2
setting=6

for gating in 0.01 0.1 0.5 1 2 5
do
    python finetune_gp.py --device $device --setting $setting --gating $gating
done
