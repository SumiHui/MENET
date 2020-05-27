# -*- coding: utf-8 -*-
# @File    : derain_wgan_tf/configuration.py
# @Info    : @ TSMC-SIGGRAPH, 2019/5/29
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.


import argparse

parser = argparse.ArgumentParser()

# Model specification
parser.add_argument("--in_channel", type=int, default=3)
parser.add_argument("--n_feats", type=int, default=32)
parser.add_argument("--num_of_down_scale", type=int, default=2)
parser.add_argument("--gen_resblocks", type=int, default=6)
parser.add_argument("--discrim_blocks", type=int, default=3)
parser.add_argument("--model_name", type=str, default="shallow_edge_lossbalance", help="deep_new_ca_edge_gram_gradbalance/deep_new_ca_edge_gram_lossbalance")

# Data specification
parser.add_argument('--original_image_dir', type=str, default="/dataset/cvpr2017_derain_dataset/testing_data",
                    help='training/testing image files base dir')
parser.add_argument('--sub_dir', type=str, default="Rain100L",
                    help='training_data{RainTrainL, RainTrainH}, testing_data{Rain100L,Rain100H}')
parser.add_argument('--blend_mode', type=str, default="linear", help='`linear` or `screen`')
parser.add_argument('--crop_size', type=int, default=224, help='')
parser.add_argument('--horizontal_flip', type=bool, default=True, help='')

# Training or test specification
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--decay_epochs', type=int, default=40, help='')
parser.add_argument('--decay_factor', type=float, default=1e-1, help='learning rate decay factor')
parser.add_argument('--num_examples_per_epoch', type=int, default=1e5,
                    help='Number of examples per epoch of training dataset')
parser.add_argument('--vgg_dir', type=str, default="/dataset/pretrained_model",
                    help='dir of vgg pre-trained params file')
parser.add_argument('--critic_updates', type=int, default=5,
                    help='Number of updates of critic')

parser.add_argument('--max_checkpoints_to_keep', type=int, default=1, help='')
parser.add_argument('--num_steps_per_display', type=int, default=10, help='')

parser.add_argument('--train_dir', type=str, default="/dataset/derain_h5", help=' h5py format dataset directory.')
parser.add_argument('--test_dir', type=str, default="/dataset/derain_h5", help='')
parser.add_argument('--data_filename', type=str, default="Rain100L.h5", help=' h5py format train/test dataset file name.')

parser.add_argument('--tensorboard', type=str, default="tensorboard", help='')
parser.add_argument('--model_dir', type=str, default="model_params", help='')
parser.add_argument('--gpu_id', type=str, default="0", help='')
parser.add_argument('--metric_dir', type=str, default="metric", help='')

parser.add_argument('--infer_in_dir', type=str, default="img/examples", help='')
parser.add_argument('--infer_out_dir', type=str, default="img/results", help='')
parser.add_argument('--scale_ratio', type=int, default=16, help='down sampling scale ratio, for inference image resize')
parser.add_argument('--ext', type=str, default=".png", help='`.jpg` or `.png`. In the inference stage, the extension of the picture')

args = parser.parse_args()


class ModelConfig(object):
    """Wrapper class for configuring model parameters."""

    def __init__(self):
        self.in_channel = args.in_channel
        self.n_feats = args.n_feats
        self.num_of_down_scale = args.num_of_down_scale
        self.gen_resblocks = args.gen_resblocks
        self.discrim_blocks = args.discrim_blocks
        self.model_name = args.model_name

        self.original_image_dir = args.original_image_dir
        self.sub_dir = args.sub_dir
        self.blend_mode = args.blend_mode
        self.crop_size = args.crop_size
        self.horizontal_flip = args.horizontal_flip

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.decay_epochs = args.decay_epochs
        self.decay_factor = args.decay_factor
        self.num_examples_per_epoch = args.num_examples_per_epoch
        self.vgg_dir = args.vgg_dir
        self.critic_updates = args.critic_updates

        self.max_checkpoints_to_keep = args.max_checkpoints_to_keep
        self.num_steps_per_display = args.num_steps_per_display

        self.train_dir = args.train_dir
        self.test_dir = args.test_dir
        self.data_filename = args.data_filename

        self.tensorboard = args.tensorboard
        self.model_dir = args.model_dir
        self.gpu_id = args.gpu_id
        self.metric_dir = args.metric_dir
        self.infer_in_dir = args.infer_in_dir
        self.infer_out_dir = args.infer_out_dir
        self.scale_ratio = args.scale_ratio
        self.ext = args.ext


cfg = ModelConfig()

if __name__ == '__main__':
    for name in args.__dict__:
        print("self.{}=args.{}".format(name, name))
