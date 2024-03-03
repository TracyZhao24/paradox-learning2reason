python logic_bert/evaluate.py \
	--data_file DATA/LP/prop_examples.balanced_by_backward.max_6.json \
	--log_file logs/eval_lp.txt \
	--device cuda --cuda_core 1 \
	--vocab_file sample/vocab.txt
