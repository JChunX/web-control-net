# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md
import tempfile
from pathlib import Path
import gzip, argparse, math, re
from functools import lru_cache
from collections import namedtuple

from tqdm import tqdm
from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from tinygrad.helpers import dtypes, GlobalCounters, Timing, Context, getenv
from tinygrad.nn import Conv2d, Linear, GroupNorm, LayerNorm, Embedding
from extra.utils import download_file
from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict
from tinygrad.jit import TinyJit

