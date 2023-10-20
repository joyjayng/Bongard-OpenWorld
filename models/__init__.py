# lift `make` to `models.make`
# register all modules
from . import bongard_model, misc
from .encoder import relation_net, swin_transformer, timm
from .head import caption_head, meta_baseline, metaoptnet, snail, wren
from .model import make
