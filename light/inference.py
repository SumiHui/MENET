# -*- coding: utf-8 -*-
# @File    : derain_wgan_tf/inference.py
# @Info    : @ TSMC-SIGGRAPH, 2019/5/30
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.


import os
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

from configuration import cfg
from utils import inference_wrapper

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id  # only /gpu:gpu_id is visible


def main(_):
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(os.path.join(cfg.model_dir, model.nickname))
    g.finalize()
    print("Restore model from directory: {}".format(os.path.join(cfg.model_dir, model.nickname)))
    filenames = list(filter(lambda x: x.endswith('.jpg'), os.listdir(cfg.infer_in_dir)))
    filenames = [os.path.join(cfg.infer_in_dir, filename) for filename in filenames]
    print("Running de-rain infer on %d files from directory: %s" % (len(filenames), cfg.infer_in_dir))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=g, config=config) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        if not os.path.exists(cfg.infer_out_dir):
            os.makedirs(cfg.infer_out_dir)
        for i, filename in enumerate(filenames):
            bgr = cv2.imread(filename)
            h, w = bgr.shape[:2]
            if w % cfg.scale_ratio != 0 or h % cfg.scale_ratio != 0:
                aw = (cfg.scale_ratio - w % cfg.scale_ratio) % cfg.scale_ratio
                ah = (cfg.scale_ratio - h % cfg.scale_ratio) % cfg.scale_ratio
                bgr = cv2.resize(bgr, (w + aw, h + ah), interpolation=cv2.INTER_CUBIC)

            rgb_array = np.expand_dims(np.asarray(bgr[..., ::-1], "float32"), 0)
            rgb_array = model.inference_step(sess=sess, input_feed=rgb_array)[0]

            basename = os.path.basename(filename).split(".")[0]
            b_output = cv2.resize(rgb_array[..., ::-1], (w, h), interpolation=cv2.INTER_CUBIC)
            print(basename, b_output.shape, np.max(b_output),np.min(b_output),np.mean(b_output))

            cv2.imwrite(os.path.join(cfg.infer_out_dir,
                                     "{}@{}_{}.png".format(basename, model.nickname, datetime.now().date())), b_output)


if __name__ == "__main__":
    tf.app.run()
