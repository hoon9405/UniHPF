CUDA_VISIBLE_DEVICES=5 \
python3 ../../train.py \
--train_task note \
--pred_src mimic3 \
--structure hi \
--pred_model transformer \
--pred_pooling mean \
--n_layers 2 \
--max_seq_len 256 \
--batch_size 16 \
--world_size 4 \
--criterion cross_entropy \
--valid_subset valid,test \
--wandb \
--best_checkpoint_metric acc \
--maximize_best_checkpoint_metric
