CUDA_VISIBLE_DEVICES=3,4,5,7 \
python3 ../../train.py \
--input_path /home/edlab/jykim/UniHPF_pretrain/input \
--train_task pretrain \
--pretrain_src mimiciii \
--pretrain_task simclr \
--structure hi \
--pred_model eventaggregator \
--pred_pooling mean \
--n_layers 2 \
--embed_dim 128 \
--max_seq_len 256 \
--batch_size 16 \
--world_size 4 \
--criterion simclr \
--model unihpf_simclr \
--time_embed aggregator \
--wandb

CUDA_VISIBLE_DEVICES=3,4,5,7 \
python3 ../../train.py \
--input_path /home/edlab/jykim/UniHPF_pretrain/input \
--train_task finetune \
--ft_pt_src mimiciii \
--ft_pt_task simclr \
--pred_src mimiciii \
--structure hi \
--pred_model eventaggregator \
--pred_pooling mean \
--n_layers 2 \
--max_seq_len 256 \
--batch_size 16 \
--world_size 4 \
--criterion prediction \
--valid_subset valid,test \
--seed 42 \
--time_embed aggregator \
--best_checkpoint_metric auroc \
--maximize_best_checkpoint_metric \
--wandb


##############################################

# CUDA_VISIBLE_DEVICES=3,4,5,7 \
# python3 ../../train.py \
# --input_path /home/edlab/jykim/UniHPF_pretrain/input \
# --train_task scratch \
# --pred_src mimiciii \
# --structure hi \
# --pred_model eventaggregator \
# --pred_pooling mean \
# --n_layers 2 \
# --max_seq_len 256 \
# --batch_size 16 \
# --world_size 4 \
# --criterion prediction \
# --valid_subset valid,test \
# --seed 42 \
# --time_embed aggregator \
# --wandb \
# --best_checkpoint_metric auroc \
# --maximize_best_checkpoint_metric

# CUDA_VISIBLE_DEVICES=3,4,5,7 \
# python3 ../../train.py \
# --input_path /home/edlab/jykim/UniHPF_pretrain/input \
# --train_task scratch \
# --pred_src mimiciii \
# --structure hi \
# --pred_model eventaggregator \
# --pred_pooling mean \
# --n_layers 2 \
# --max_seq_len 256 \
# --batch_size 16 \
# --world_size 4 \
# --criterion prediction \
# --valid_subset valid,test \
# --seed 42 \
# --wandb \
# --best_checkpoint_metric auroc \
# --maximize_best_checkpoint_metric



# --best_checkpoint_metric auprc \
# --maximize_best_checkpoint_metric

# device_ids=$1
# pred_source=$2
# batch=$3
# world=$4

# tasks="im_disch fi_ac"
# for task in $tasks
# do
#     CUDA_VISIBLE_DEVICES=$device_ids\
#     python3 ../../train.py \
#     --train_task scratch \
#     --pred_src $pred_source \
#     --structure hi \
#     --pred_model transformer \
#     --pred_pooling mean \
#     --n_layers 2 \
#     --max_seq_len 256 \
#     --batch_size $batch \
#     --world_size $world \
#     --criterion cross_entropy \
#     --valid_subset valid,test \
#     --pred_target $task \
#     --wandb \
#     --best_checkpoint_metric auprc \
#     --maximize_best_checkpoint_metric
# done

######################################## For encoder embedding finetune ########################################
# CUDA_VISIBLE_DEVICES=3,4,5,7 \
# python3 ../../train.py \
# --input_path /home/edlab/jykim/UniHPF_pretrain/input \
# --train_task pretrain \
# --pretrain_src mimiciii \
# --pretrain_task simclr \
# --structure hi \
# --pred_model eventaggregator \
# --pred_pooling mean \
# --n_layers 2 \
# --embed_dim 128 \
# --max_seq_len 256 \
# --batch_size 16 \
# --world_size 4 \
# --criterion simclr \
# --model unihpf_simclr \
# --time_embed encoder \
# --wandb

# CUDA_VISIBLE_DEVICES=3,4,5,7 \
# python3 ../../train.py \
# --input_path /home/edlab/jykim/UniHPF_pretrain/input \
# --train_task finetune \
# --ft_pt_src mimiciii \
# --ft_pt_task simclr \
# --pred_src mimiciii \
# --structure hi \
# --pred_model eventaggregator \
# --pred_pooling mean \
# --n_layers 2 \
# --max_seq_len 256 \
# --batch_size 16 \
# --world_size 4 \
# --criterion prediction \
# --valid_subset valid,test \
# --seed 42 \
# --time_embed encoder \
# --best_checkpoint_metric auroc \
# --maximize_best_checkpoint_metric \
# --wandb