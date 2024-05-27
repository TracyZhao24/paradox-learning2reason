python3 logic_bert/train_logic_bert.py \
                --data_file DATA/LP/prop_examples.balanced_by_backward.max_6.json \
                --other_dist_data_file DATA/RP/prop_examples.balanced_by_backward.max_6.json \
                --device cuda --cuda_core 2 \
                --max_epoch 40 --batch_size 1 --effective_batch_size 1024 --lr 0.0002 --weight_decay 0.001 \
                --vocab_file sample/vocab.txt \
                --max_reasoning_depth 3 \
                --model_layers 7 \
                --experiment_directory EXPERIMENTS/22_may_2024/