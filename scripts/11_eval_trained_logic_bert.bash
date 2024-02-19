python logic_bert/eval_trained_logic_bert.py \
                --data_file DATA/RP/prop_examples.balanced_by_backward.max_6.json \
                --device cuda --cuda_core 1 --max_epoch 20 \
                --batch_size 64 --lr 0.00004 --weight_decay 0.001 \
                --log_file log.txt --output_model_file OUTPUT/LP/LOGIC_BERT/model.pt \
                --vocab_file sample/vocab.txt
