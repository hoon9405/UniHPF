device_ids=$1
pred_source=$2
batch=$3
world=$4

tasks="fi_ac im_disch"

for task in $tasks
do
   CUDA_VISIBLE_DEVICES=$device_ids \
   OMP_NUM_THREADS=8 \
   python ../../train.py \
   --train_task=scratch \
   --pred_src $pred_source \
   --structure fl \
   --pred_model=performer \
   --pred_pooling mean \
   --max_seq_len=8192 \
   --batch_size $batch \
   --world_size $world \
   --criterion=cross_entropy \
   --valid_subset=valid,test \
   --pred_target=$task \
   --wandb \
   --best_checkpoint_metric auprc \
   --maximize_best_checkpoint_metric
done

tasks="mort readm los3 los7 dx"

for task in $tasks
do
   CUDA_VISIBLE_DEVICES=$device_ids \
   OMP_NUM_THREADS=8 \
   python ../../train.py \
   --train_task=scratch \
   --pred_src $pred_source \
   --structure fl \
   --pred_model=performer \
   --pred_pooling mean \
   --max_seq_len=8192 \
   --batch_size $batch \
   --world_size $world \
   --criterion=binary_cross_entropy \
   --valid_subset=valid,test \
   --pred_target=$task \
   --wandb \
   --best_checkpoint_metric auprc \
   --maximize_best_checkpoint_metric
done

