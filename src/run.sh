#!/bin/sh
python sdr.py --train --embed_dropout 0.5 --lr 0.0005 --model lingunet --bidirectional True --num_lingunet_layers 2 --model_save --run_name sdr_trial_4 



#!/bin/sh
python sdr.py --train --embed_dropout 0.5 --lr 0.00001 --linear_hidden_size 512 --model clip --bidirectional True --num_lingunet_layers 2 --model_save --run_name sdr_trial_7_2


nohup python sdr.py --train --annotate_regions --box_restriction top500 --embed_dropout 0.5 --lr 0.00001 --linear_hidden_size 512 --model clip --bidirectional True --num_lingunet_layers 2 --model_save --run_name higher_alpha_prompt_tint

CUDA_VISIBLE_DEVICES=6,7 nohup python sdr.py --train --annotate_objects --box_restriction top500 --embed_dropout 0.5 --lr 0.00001 --linear_hidden_size 512 --model clip --bidirectional True --num_lingunet_layers 2 --model_save --run_name object_annotation


python sdr.py --evaluate --embed_dropout 0.5 --lr 0.00001 --linear_hidden_size 512 --model clip --bidirectional True --num_lingunet_layers 2 --eval_ckpt /data2/saaket/models/checkpoints/sdr_trial_7_2/best_model.pt --run_name sdr_test_clip_lunet_8to1



python sdr.py --evaluate --embed_dropout 0.5 --lr 0.00001 --model lingunet --bidirectional True --num_lingunet_layers 2 --eval_ckpt /data2/saaket/models/checkpoints/sdr_trial_4/best_model_epoch_15_0.25021864067966015.pt --run_name sdr_test_clip_lunet_8to1