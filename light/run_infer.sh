#!/bin/bash

infer_in_dir="img/examples"
infer_out_dir="img/results"
scale_ratio=16
gpu_id=1
extension=".jpg"


# models=("menet_shallow_new" \
#         "menet_shallow_new_edge_fixed" "menet_shallow_new_edge_gradbalance" "menet_shallow_new_edge_lossbalance" \
#         "menet_shallow_new_edge_gram_lossbalance" "menet_shallow_new_edge_gram_gradbalance" \
#         "menet_shallow_new_ca_edge_gram_lossbalance" "menet_shallow_new_ca_edge_gram_gradbalance"
#         "menet_deep_new_ca_edge_gram_lossbalance" "menet_deep_new_ca_edge_gram_gradbalance")

models=("menet_shallow_new_ca_edge_gram_lossbalance")

for model in ${models[*]}
do
    python -u inference.py \
        --model_name ${model} \
        --infer_in_dir ${infer_in_dir} \
        --infer_out_dir ${infer_out_dir} \
        --scale_ratio ${scale_ratio} \
        --ext ${extension} \
        --gpu_id ${gpu_id}
done
