# required by (\ SHELL COMMANDS \) #269
SHELL:=/bin/bash

.PHONY: train
train:
	( \
	 python trainer_sampler_data_uni.py \
 			--output_dir ./models/test \
 			--overwrite_output_dir \
 			--save_steps 10000 \
 			--seed 1029 \
 			--num_train_epochs 15 \
 			--learning_rate 5e-5 \
 			--per_gpu_train_batch_size 16 \
 			--gradient_accumulation_steps 1 \
 			--per_gpu_eval_batch_size 32 \
 			--max_seq_len 128 \
 			--dynamic_padding \
 			--smart_batching \
 			--fp16 \
 			--num_labels 273 \
			--multilabel \
			--logging_steps 1646 \
 			--evaluate_during_training ; \
	)

.PHONY: inference
inference:
	( \
	 python inference.py \
	 		--output_dir ./models/test \
	 		--model_name_or_path ./models/test \
 			--seed 1029 \
 			--per_gpu_eval_batch_size 32 \
 			--max_seq_len 128 \
 			--dynamic_padding \
 			--smart_batching \
			--multilabel \
			--num_labels 273 \
 			--fp16 ; \
	)
