#!/bin/bash
train_dir="/dataset/cvpr2017_derain_dataset/training_data"
test_dir="/dataset/cvpr2017_derain_dataset/testing_data"
train_heavy_sub_dir="RainTrainH"
train_light_sub_dir="RainTrainL"
test_heavy_sub_dir="Rain100H"
test_light_sub_dir="Rain100L"

python3 build_h5_dataset.py --base_dir ${train_dir} --sub_dir ${train_heavy_sub_dir}
python3 build_h5_dataset.py --base_dir ${train_dir} --sub_dir ${train_light_sub_dir}
python3 build_h5_dataset.py --base_dir ${test_dir} --sub_dir ${test_heavy_sub_dir}
python3 build_h5_dataset.py --base_dir ${test_dir} --sub_dir ${test_light_sub_dir}