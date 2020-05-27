#!/bin/bash
DATE=$(date +%Y%m%d%H%M)
logdir="log/"

batch_size=20
lr=1e-3
epochs=100
decay_epochs=40
num_examples_per_epoch=18000
train_dir="dataset/derain_h5"
data_filename="RainTrainH.h5"
vgg_dir="dataset/pretrained_models"
gpu_id=1

if [ -d ${logdir} ]; then
    # echo "Detects folder exists"
    break
else
    mkdir -v -p ${logdir}
fi

# models=("menet_shallow_new" \
#         "menet_shallow_new_edge_fixed" "menet_shallow_new_edge_gradbalance" "menet_shallow_new_edge_lossbalance" \
#         "menet_shallow_new_edge_gram_lossbalance" "menet_shallow_new_edge_gram_gradbalance" \
#         "menet_shallow_new_ca_edge_gram_lossbalance" "menet_shallow_new_ca_edge_gram_gradbalance"
#         "menet_deep_new_ca_edge_gram_lossbalance" "menet_deep_new_ca_edge_gram_gradbalance")

models=("menet_shallow_new_ca_edge_gram_lossbalance")

for model in ${models[*]}
do
    nohup python3 -u train.py \
        --model_name ${model} \
        --batch_size ${batch_size} \
        --num_examples_per_epoch ${num_examples_per_epoch} \
        --lr ${lr} \
        --epochs ${epochs} \
        --decay_epochs ${decay_epochs} \
        --train_dir ${train_dir} \
        --data_filename ${data_filename} \
        --vgg_dir ${vgg_dir} \
        --gpu_id ${gpu_id} > "${logdir}${model}.log.$DATE" &
done
