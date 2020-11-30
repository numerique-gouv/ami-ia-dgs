# required by (\ SHELL COMMANDS \)
SHELL:=/bin/bash

.PHONY: train
train:
	( \
	 python trainer_sampler.py \
 			--output_dir ./models/test \
 			--overwrite_output_dir \
 			--save_steps 10000 \
 			--seed 1029 \
 			--num_train_epochs 5 \
 			--learning_rate 5e-5 \
 			--per_gpu_train_batch_size 16 \
 			--gradient_accumulation_steps 1 \
 			--per_gpu_eval_batch_size 16 \
 			--max_seq_len 256 \
 			--dynamic_padding \
 			--smart_batching \
 			--fp16 \
 			--num_labels 173 \
			--class_weights \
 			--evaluate_during_training ; \
	)

.PHONY: inference
inference:
	( \
	 python inference.py \
	 		--output_dir ./models/test \
	 		--do_predict True \
			--do_train False \
	 		--model_name_or_path ./models/test \
 			--seed 1029 \
 			--per_gpu_eval_batch_size 16 \
 			--max_seq_len 512 \
 			--dynamic_padding \
 			--smart_batching \
 			--fp16 ; \
	)
