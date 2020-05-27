# -*- coding: utf-8 -*-
# @File    : MENET/net.py
# @Info    : @ TSMC-SIGGRAPH, 2019/8/10
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.

from configuration import cfg

from template import menet_shallow_new, menet_shallow_new_ca
from template import menet_shallow_new_edge_lossbalance, menet_shallow_new_edge_gradbalance, menet_shallow_new_edge_fixed

from template import menet_shallow_new_edge_gram_gradbalance, menet_shallow_new_edge_gram_lossbalance

from template import menet_shallow_new_ca, menet_shallow_new_ca_edge_gram_gradbalance, menet_shallow_new_ca_edge_gram_lossbalance
from template import menet_deep_new_ca_edge_gram_gradbalance, menet_deep_new_ca_edge_gram_lossbalance

from template import menet_shallow_new_vgg_fixed

models = {
    "menet_shallow_new": menet_shallow_new.ModelShallowNew,

    "menet_shallow_new_edge_fixed": menet_shallow_new_edge_fixed.ModelShallowNewEdgeFixed,
    "menet_shallow_new_edge_lossbalance": menet_shallow_new_edge_lossbalance.ModelShallowNewEdgeLossBalance,
    "menet_shallow_new_edge_gradbalance": menet_shallow_new_edge_gradbalance.ModelShallowNewEdgeGradBalance,
    "menet_shallow_new_edge_gram_lossbalance":menet_shallow_new_edge_gram_lossbalance.ModelShallowNewEdgeGramLossBalance,
    "menet_shallow_new_edge_gram_gradbalance": menet_shallow_new_edge_gram_gradbalance.ModelShallowNewEdgeGramGradBalance,

    "menet_shallow_new_ca": menet_shallow_new_ca.ModelShallowNewCa,
    "menet_shallow_new_ca_edge_gram_lossbalance": menet_shallow_new_ca_edge_gram_lossbalance.ModelShallowNewCaEdgeGramLossBalance,
    "menet_shallow_new_ca_edge_gram_gradbalance": menet_shallow_new_ca_edge_gram_gradbalance.ModelShallowNewCaEdgeGramGradBalance,
    "menet_deep_new_ca_edge_gram_lossbalance": menet_deep_new_ca_edge_gram_lossbalance.ModelDeepNewCaEdgeGramLossBalance,
    "menet_deep_new_ca_edge_gram_gradbalance": menet_deep_new_ca_edge_gram_gradbalance.ModelDeepNewCaEdgeGramGradBalance,

    "menet_shallow_new_vgg_fixed": menet_shallow_new_vgg_fixed.ModelShallowNewVGGFixed,
}

Model = models[cfg.model_name]
