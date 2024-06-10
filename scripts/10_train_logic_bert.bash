python3 logic_bert/train_logic_bert.py \
                --data_file DATA/LP/prop_examples.balanced_by_backward.max_6.json \
                --other_dist_data_file DATA/RP/prop_examples.balanced_by_backward.max_6.json \
                --device cuda --cuda_core 1 \
                --max_epoch 100 --batch_size 1 --effective_batch_size 1024 --lr 0.0002 --weight_decay 0.001 \
                --vocab_file sample/vocab.txt \
                --max_reasoning_depth 5 \
                --model_layers 8 \
                --optimizer SGD \
                --experiment_directory EXPERIMENTS/10_june_2024_SGD_not_Adam/