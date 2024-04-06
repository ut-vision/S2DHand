

#python3 train_detnet_dual.py -trs ah -tes ah --evaluate -eid 37 --gpus 0 --pic -1 --checkpoint checkpoints0 --setup 0 --pair 0,3
#for setup in 0 1
#do
#  for pair in 0,1 #0,2 0,3 1,2 1,3 2,3
#  do
#    python3 train_detnet_dual.py -trs ah -tes ah --pic 1024 --root_idx 0 --resume -eid 37 --epochs 10 --start_epoch 1 --gpus 0 --checkpoint in_dataset_adapt --setup ${setup} --pair ${pair}
#  done
#done

for setup in 0 #1
do
  for pair in 1,2 #0,2 0,3 1,2 1,3 2,3
  do
    python3 train_detnet_dual.py -trs ah -tes ah --root_idx 9 --pic 1024 --resume -eid 68 --epochs 10 --start_epoch 1 --gpus 0 --checkpoint cross_dataset_adapt --setup ${setup} --pair ${pair}
  done
done

