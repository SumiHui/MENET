# -*- coding: utf-8 -*-
# @File    : derain_wgan_tf/inference_wrapper.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/12
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


from net import Model
from utils import inference_wrapper_base


class InferenceWrapper(inference_wrapper_base.InferenceWrapperBase):
    """Model wrapper class for performing inference with a ShowAndTellModel."""

    def __init__(self):
        super(InferenceWrapper, self).__init__()
        self.nickname = self.build_model().nickname

    def build_model(self):
        model = Model(mode="inference")
        model.build()
        return model

    # def inference_step(self, sess, input_feed, img_size_feed):
    #     b_output, r_output = sess.run(
    #         fetches=["derain/bg_hat:0", "derain/r_hat:0"],
    #         feed_dict={
    #             "image_feed:0": input_feed,
    #             "img_size_feed:0": img_size_feed,
    #         })
    #     return b_output, r_output

    def inference_step(self, sess, input_feed):
        output = sess.run(
            fetches="derain/output:0",
            feed_dict={
                "image_feed:0": input_feed
            })
        return output
