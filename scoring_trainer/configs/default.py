# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN(new_allowed=True)


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  return _C.clone()

