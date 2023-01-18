device_num=$1
world_size=$2
train_src=$3
pred_task=$4

OMP_NUM_THREADS=16 \
CUDA_VISIBLE_DEVICES=$device_num \
python3 ../../main.py \
--train_task scratch \
--train_src $train_src \
--structure hi \
--emb_type textbase \
--feature whole \
--pred_model eventaggregator \
--n_layers 2 \
--max_seq_len 256 \
--batch_size 16 \
--world_size $world_size \
--criterion prediction \
--valid_subset valid,test \
--pred_task $pred_task \
--wandb \
# --pooled_eval \