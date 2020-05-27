### Dataset prepare
Dataset from CVPR2017 paper **W. Yang, R. T. Tan, J. Feng, J. Liu, Z. Guo, and S. Yan, "Deep Joint Rain Detection and Removal
from a Single Image," in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017**

Down load from URL: http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html
### Data preprocess: build dataset
```bash
bash run_build_h5_dataset.sh
```

### Training model on default dataset (RainTrainH)
```bash
nohup python3 -u train.py > log &
```



用户手册:
#### 程序的使用(训练和测试)

将本程序至于任何你想放置的位置，例如`/home/sumihui/`,以下说明将使用$HOME代指本程序项目根目录，以程序放置于`/home/sumihui/`为例，
则`$HOME="/home/sumihui/rain_detection_and_removal"`

##### 数据集下载
从`http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html`下载训练和测试数据集(Rain100H,Rain100L;
RainTrainH,RainTrainL一共四个数据集，前两个为测试集，后两个为训练集),并解压到路径`/dataset/cvpr2017_derain_dataset`，
其中，测试集解压到`/dataset/cvpr2017_derain_dataset/testing_data`,训练集解压到`/dataset/cvpr2017_derain_dataset/training_data`.

 * 解压后数据集的目录结构如下(/dataset/cvpr2017_derain_dataset)：
```
.
├── testing_data
│   ├── Rain100H
│   └── Rain100L
└── training_data
       ├── RainTrainH
       └── RainTrainL
```

 * 在命令窗口执行以下命令完成数据加工。该命令将对原始数据集进行合成，裁剪，翻转，并输出h5py文件到默认目录`/dataset/derain_h5`。
```bash
bash run_build_h5_dataset.sh
```
[注]程序默认加工线性叠加模型的数据，如需更换模型，在`configuration.py`中修改`mode`值即可。
```

程序目录结构
```bash
.
├── build_h5_dataset.py
├── configuration.py
├── data_helper.py
├── dataset
│   ├── derain_h5 -> /home/tsmc/sumihui/dataset/derain_h5/
│   └── pretrained_models -> /home/tsmc/sumihui/dataset/pretrained_models/
├── img
│   ├── examples
│   └── results
├── inference.py
├── log
│   ├── shallow_edge_fixed.log.201912201030
│   ├── ......
│   └── shallow.log.201912201013
├── metric
│   └── metrics.txt
├── model_params
│   ├── ModelShallow
│   ├── ......
│   └── ModelShallowSPAEdgeLossBalance
├── net.py
├── readme.md
├── run_build_h5_dataset.sh
├── run_test.sh
├── run_train.sh
├── template
│   ├── __init__.py
│   ├── net_base.py
│   ├── ......
│   └── net_shallow.py
├── train.py
├── utils
│   ├── inference_wrapper_base.py
│   ├── inference_wrapper.py
│   ├── __init__.py
│   └── transforms.py
├── validation.py
└── vgg19.py
```