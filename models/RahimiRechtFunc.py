import torch.nn as nn
import torch
import math


class RahimiRechtFuncModule(nn.Module):
  
  """Ill-conditioned linear regression problem, with a linear net (default 2 layers).
  By Rahimi & Recht, from: http://www.argmin.net/2017/12/05/kitchen-sinks/
  Unofficial implementation
  """
  def __init__(self, xdim = 6, wdim= 6, ydim = 10, nsamples = 1000, A_condition_number = 1e-5, num_layers=2):
      super(RahimiRechtFuncModule, self).__init__()

      self.Atrue = torch.linspace(start=1, end=A_condition_number, steps=ydim).reshape(-1, 1)*torch.rand(ydim, xdim)
      self.X = torch.randn(nsamples, xdim)
      Ytrue = self.Atrue@self.X.T
      self.Ytrue = Ytrue.T

      num_dims = wdim * xdim + wdim * wdim * (num_layers - 2) + ydim * wdim

      (self.nsamples, self.num_layers, self.xdim, self.wdim, self.ydim, self.cond_num) = (nsamples, num_layers, xdim, wdim, ydim, A_condition_number)
      self.layers = nn.Sequential(
          nn.Linear(self.xdim, self.wdim, bias = False)
      )


      for _ in range(self.num_layers-2):
          self.layers.append(nn.Linear(self.wdim, self.wdim, bias=False))
      self.layers.append(nn.Linear(self.wdim, self.ydim, bias=False))

      for idx in range(self.num_layers):
          nn.init.normal_(self.layers[idx].weight)
          if num_layers >2:
              s =math.sqrt(wdim)
              self.layers[idx].weight.data.div_(s)

  def forward(self, X):
      return self.layers(X)

  def __str__(self):
      name = "dim: {}x{}x{}, nsample: {}, nlayers: {}, cond_num: {}".format(self.xdim, self.wdim, self.ydim, self.nsamples, self.num_layers, self.cond_num)
      return name
  
  def compute_ytrue(self, X):
      y = self.Atrue @ X.t()
      return y.t()
