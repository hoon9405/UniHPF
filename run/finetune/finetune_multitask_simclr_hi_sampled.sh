device_num=$1
world_size=$2
train_src=$3
pt_src=$4
batch_size=$5
seed=$6

OMP_NUM_THREADS=16 \
CUDA_VISIBLE_DEVICES=$device_num \
python3 ../../main.py \
--train_task finetune \
--pretrain_task simclr \
--train_src $train_src \
--pt_src $pt_src \
--structure hi \
--emb_type textbase \
--feature whole \
--pred_model eventaggregator \
--n_layers 2 \
--max_seq_len 256 \
--batch_size $batch_size \
--world_size $world_size \
--criterion prediction \
--best_checkpoint_metric avg_auroc \
--maximize_best_checkpoint_metric \
--valid_subset valid,test \
--pretrain_sample sampled \
--seed $seed \
--wandb \
