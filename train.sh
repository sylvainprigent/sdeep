#!/bin/sh

#sdtrain --model DnCNN --loss MSELoss --optim Adam \
#        --train_dataset RestorationPatchDataset --val_dataset RestorationDataset \
#        --workflow SWorkflow --save ./runs/run_7 --train_batch_size 128 --val_batch_size 3 \
#        --rpd_path_source /home/sprigent/Documents/datasets/airbus/original_250_small/train/noise120/ \
#        --rpd_path_target /home/sprigent/Documents/datasets/airbus/original_250_small/train/GT/ \
#        --rd_path_source /home/sprigent/Documents/datasets/airbus/original_250/test/noise120/ \
#        --rd_path_target /home/sprigent/Documents/datasets/airbus/original_250/test/GT/ \
#        --dncnn_layers 4 \
#        --lr 0.001 \
#        --epochs 50 \
#        --reuse=true


sdtrain --model DnCNN --loss MSELoss --optim Adam \
        --train_dataset RestorationPatchDataset --val_dataset RestorationDataset \
        --workflow SWorkflow --save ./runs --train_batch_size 128 --val_batch_size 4 \
        --rpd_path_source /home/sprigent/Documents/datasets/simulation/simulated/train/B120/ \
        --rpd_path_target /home/sprigent/Documents/datasets/simulation/simulated/train/GT/ \
        --rd_path_source /home/sprigent/Documents/datasets/simulation/simulated/test/B120/ \
        --rd_path_target /home/sprigent/Documents/datasets/simulation/simulated/test/GT/ \
        --dncnn_layers 17 \
        --lr 0.001 \
        --epochs 50 \
        --tiling True