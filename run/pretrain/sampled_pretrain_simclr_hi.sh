
device_num=$1
world_size=$2
train_src=$3
batch_size=$4

OMP_NUM_THREADS=16 \
CUDA_VISIBLE_DEVICES=$device_num \
python3 ../../main.py \
--train_task sampled_pretrain \
--train_src $train_src \
--pretrain_task simclr \
--structure hi \
--pred_model eventaggregator \
--pred_pooling mean \
--n_layers 2 \
--embed_dim 128 \
--max_seq_len 256 \
--batch_size $batch_size \
--world_size $world_size \
--criterion simclr \
--model unihpf_simclr \
--emb_type textbase \
--feature whole \
--valid_subset "" \
--best_checkpoint_metric loss \
--maximize_best_checkpoint_metric \
--wandb