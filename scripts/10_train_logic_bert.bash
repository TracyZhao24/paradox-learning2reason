python logic_bert/train_logic_bert.py \
                --data_file DATA/LP/prop_examples.balanced_by_backward.max_6.json \
                --device cuda --cuda_core 1 --max_epoch 20 \
                --batch_size 1 --lr 0.00004 --weight_decay 0.001 \
                --loss_log_file logs/LP/loss_log_3.txt --acc_log_file logs/LP/acc_log_3.txt \
                --output_model_file OUTPUT/LP/LOGIC_BERT/model_3.pt \
                --word_emb_file OUTPUT/LP/LOGIC_BERT/word_emb_3.pt \
                --position_emb_file OUTPUT/LP/LOGIC_BERT/position_emb_3.pt \
                --vocab_file sample/vocab.txt
