#!/bin/bash

# evaluate by example depth
for ((i = 0; i < 4; i++)); do
    python logic_bert/eval_trained_logic_bert.py \
                    --data_file DATA/RP/prop_examples.balanced_by_backward.max_6.json \
                    --depth "$i" \
                    --model_path /space/trzhao/paradox-learning2reason/OUTPUT/LP/LOGIC_BERT/model_3.pt \
                    --word_emb_path OUTPUT/LP/LOGIC_BERT/word_emb_3.pt \
                    --position_emb_path OUTPUT/LP/LOGIC_BERT/position_emb_3.pt \
                    --device cuda --cuda_core 0 \
                    --batch_size 128 \
                    --log_file logs/eval_trained_logic_bert_lp.txt \
                    --vocab_file sample/vocab.txt 
done