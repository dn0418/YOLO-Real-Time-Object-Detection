from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):
  """
  Takes a configuration file
    
  Returns a list of blocks. Each blocks describes a block in the neural
  network to be built. Block is represented as a dictionary in the list  
  """

  file = open(cfgfile, 'r')
  lines = file.read().split('\n')                        # store the lines in a list
  lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
  lines = [x for x in lines if x[0] != '#']              # get rid of comments
  lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

  block = {}
  blocks = []

  for line in lines:
    if line[0] == "[":               # This marks the start of a new block
      if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
        blocks.append(block)     # add it the blocks list
        block = {}               # re-init the block
      block["type"] = line[1:-1].rstrip()     
    else:
      key,value = line.split("=") 
      block[key.rstrip()] = value.lstrip()
  blocks.append(block)

  return blocks