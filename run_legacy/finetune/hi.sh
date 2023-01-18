device_ids=$1
pretrain_source=$2
pred_source=$3
batch=$4
world=$5

pretrain_methods="simclr w2v"
tasks="mort los3 los7 readm dx"
for mth in $pretrain_methods
do
    for task in $tasks
    do
        CUDA_VISIBLE_DEVICES=$device_ids \
        python3 ../../train.py \
        --train_task finetune \
        --ft_pt_src $pretrain_source \
        --ft_pt_task $mth \
        --pred_src $pred_source \
        --structure hi \
        --pred_model transformer \
        --pred_pooling mean \
        --n_layers 2 \
        --max_seq_len 256 \
        --batch_size $batch \
        --world_size $world \
        --criterion binary_cross_entropy \
        --valid_subset valid,test \
        --pred_target $task \
        --wandb \
        --best_checkpoint_metric auprc \
        --maximize_best_checkpoint_metric
    done
done

pretrain_methods="simclr w2v"
tasks="fi_ac im_disch"
for mth in $pretrain_methods
do
    for task in $tasks
    do
        CUDA_VISIBLE_DEVICES=$device_ids \
        python3 ../../train.py \
        --train_task finetune \
        --ft_pt_src $pretrain_source \
        --ft_pt_task $mth \
        --pred_src $pred_source \
        --structure hi \
        --pred_model transformer \
        --pred_pooling mean \
        --n_layers 2 \
        --max_seq_len 256 \
        --batch_size $batch \
        --world_size $world \
        --criterion cross_entropy \
        --valid_subset valid,test \
        --pred_target $task \
        --wandb \
        --best_checkpoint_metric auprc \
        --maximize_best_checkpoint_metric
    done
done