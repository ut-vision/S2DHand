#!/bin/bash

gpu=0

#Adaptation under in-dataset scenarios
for setup in 0 #1
do
  for pair in 1,2 #0,1 0,2 0,3 1,2 1,3 2,3
  do
    # --root_idx needs to be set as 0 for in-dataset adaptation
#    python3 adapt_detnet_dual.py -trs ah -tes ah --root_idx 0 --pic 1024 --resume -eid 37 --epochs 10 --start_epoch 1 --gpus ${gpu} --checkpoint in_dataset_adapt --setup ${setup} --pair ${pair}
    # calculate the Mono-M and Dual-M metrics
    python3 calc_metrics.py --checkpoint in_dataset_adapt/evaluation/ah --setup ${setup} --pair ${pair}
  done
done

#Adaptation under cross-dataset scenarios
for setup in 0 #1
do
  for pair in 1,2 #0,1 0,2 0,3 1,2 1,3 2,3
  do
    # --root_idx needs to be set as 9 for cross-dataset adaptation
#    python3 adapt_detnet_dual.py -trs ah -tes ah --root_idx 9 --pic 1024 --resume -eid 68 --epochs 10 --start_epoch 1 --gpus ${gpu} --checkpoint cross_dataset_adapt --setup ${setup} --pair ${pair}
    # calculate the Mono-M and Dual-M metrics
    python3 calc_metrics.py --checkpoint cross_dataset_adapt/evaluation/ah --setup ${setup} --pair ${pair}
  done
done

