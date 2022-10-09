#!/bin/sh
python sdr.py --train --embed_dropout 0.5 --lr 0.0005 --model lingunet --bidirectional True --num_lingunet_layers 2 --model_save --run_name sdr_trial_4 



#!/bin/sh
python sdr.py --train --embed_dropout 0.5 --lr 5e-5 --linear_hidden_size 512 --model clip --bidirectional True --num_lingunet_layers 2 --model_save --run_name sdr_trial_5 