#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# from abc import ABC, abstractmethod

# class Action(ABC):
#   def __init__(self, comm, src, dest):
#     self.src = src
#     self.dest = dest
#     self.comm = comm
  
#   @abstractmethod
#   def send(self):
#     pass

#   @abstractmethod
#   def receive(self):
#     pass


class SwapWeights():
  def __init__(self, src, dest):
    self.src = src
    self.dest = dest
  def __str__(self):
    return f"SwapWeights({self.src}, {self.dest})"

class ShareWeights():
  def __init__(self, src, dest):
    self.src = src
    self.dest = dest
  def __str__(self):
    return f"ShareWeights({self.src}, {self.dest})"
  
  # def send(self):
  #   self.comm.send(obj=self.global_weights, dest=self.dest, tag="ShareWeights")

  # def receive(self):
  #   global_weights = self.comm.recv(source=self.src, tag="ShareWeights")
  #   return global_weights
        
class ShareRepresentations():
  def __init__(self, src, dest, indices):
    self.src = src
    self.dest = dest
    self.indices = indices
  def __str__(self):
    return f"ShareRepresentations({self.src}, {self.dest}, {len(self.indices)} samples)"