python logic_bert/eval_trained_logic_bert.py \
                --data_file DATA/LP/prop_examples.balanced_by_backward.max_6.json \
                --device cuda --cuda_core 1 \
                --batch_size 1 \
                --log_file logs/eval_logic_bert_lp.txt \
                --vocab_file sample/vocab.txt
