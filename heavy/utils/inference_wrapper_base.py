# -*- coding: utf-8 -*-
# @File    : derain_wgan_tf/inference_wrapper_base.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/12
# @Desc    : refer to google's im2txt
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os.path

import tensorflow as tf


# pylint: disable=unused-argument


class InferenceWrapperBase(object):
    """Base wrapper class for performing inference with an image-to-text model."""

    def __init__(self):
        pass

    def build_model(self):
        """Builds the model for inference.
        Args:
          model_config: Object containing configuration for building the model.
        Returns:
          model: The model object.
        """
        tf.logging.fatal("Please implement build_model in subclass")

    def _create_restore_fn(self, checkpoint_path, saver):
        """Creates a function that restores a model from checkpoint.
        Args:
          checkpoint_path: Checkpoint file or a directory containing a checkpoint
            file.
          saver: Saver for restoring variables from the checkpoint file.
        Returns:
          restore_fn: A function such that restore_fn(sess) loads model variables
            from the checkpoint file.
        Raises:
          ValueError: If checkpoint_path does not refer to a checkpoint file or a
            directory containing a checkpoint file.
        """
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if not checkpoint_path:
                raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

        def _restore_fn(sess):
            tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
            saver.restore(sess, checkpoint_path)
            tf.logging.info("Successfully loaded checkpoint: %s",
                            os.path.basename(checkpoint_path))
            print("Successfully loaded checkpoint: ", os.path.basename(checkpoint_path))

        return _restore_fn

    def build_graph_from_config(self, checkpoint_path):
        """Builds the inference graph from a configuration object.
        Args:
          model_config: Object containing configuration for building the model.
          checkpoint_path: Checkpoint file or a directory containing a checkpoint
            file.
        Returns:
          restore_fn: A function such that restore_fn(sess) loads model variables
            from the checkpoint file.
        """
        tf.logging.info("Building model.")
        # self.build_model()    # move to inference_warpper.__init__ for get class name
        saver = tf.train.Saver()

        return self._create_restore_fn(checkpoint_path, saver)

    # def inference_step(self, sess, input_feed, img_size_feed):
    #     """Runs one step of inference.
    #     Args:
    #       sess: TensorFlow Session object.
    #       input_feed: A numpy array of shape [batch_size].
    #       img_size_feed: A list of image height and width
    #     Returns:
    #       rain_layer: A numpy array of shape [N,H,W,C].
    #       background_layer: A numpy array of shape [N,H,W,C].
    #
    #     """
    #     tf.logging.fatal("Please implement inference_step in subclass")
    def inference_step(self, sess, input_feed):
        """Runs one step of inference.
        Args:
          sess: TensorFlow Session object.
          input_feed: A numpy array of shape [batch_size].
        Returns:
          rain_layer: A numpy array of shape [N,H,W,C].
          background_layer: A numpy array of shape [N,H,W,C].
        """
        tf.logging.fatal("Please implement inference_step in subclass")
