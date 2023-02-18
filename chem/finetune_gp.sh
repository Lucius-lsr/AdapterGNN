#!/bin/bash

device=2
setting=6

for gating in 0.01 0.1 0.5 0.9 0.99 1
do
    python finetune_gp.py --device $device --setting $setting --gating $gating
done
