CUDA_VISIBLE_DEVICES=0 \
python3 ../../main.py \
--train_task scratch \
--train_src mimiciii_mimiciv \
--structure hi \
--emb_type textbase \
--feature whole \
--pred_model eventaggregator \
--n_layers 2 \
--max_seq_len 256 \
--batch_size 16 \
--world_size 1 \
--criterion prediction \
--valid_subset valid,test \
--wandb \
--pooled_eval \