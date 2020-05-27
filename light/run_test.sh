#!/bin/bash


batch_size=20
test_dir="dataset/derain_h5"
data_filename="Rain100L.h5"
vgg_dir="dataset/pretrained_models"
gpu_id=2


# models=("menet_shallow_new" \
#         "menet_shallow_new_edge_fixed" "menet_shallow_new_edge_gradbalance" "menet_shallow_new_edge_lossbalance" \
#         "menet_shallow_new_edge_gram_lossbalance" "menet_shallow_new_edge_gram_gradbalance" \
#         "menet_shallow_new_ca_edge_gram_lossbalance" "menet_shallow_new_ca_edge_gram_gradbalance"
#         "menet_deep_new_ca_edge_gram_lossbalance" "menet_deep_new_ca_edge_gram_gradbalance")

models=("menet_shallow_new_ca_edge_gram_lossbalance")

for model in ${models[*]}
do
    python3 validation.py \
        --model_name ${model} \
        --batch_size ${batch_size} \
        --test_dir ${test_dir} \
        --data_filename ${data_filename} \
        --vgg_dir ${vgg_dir} \
        --gpu_id ${gpu_id}
done
