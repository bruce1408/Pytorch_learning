import os
from ctypes import *

from ctypes import *

test = cdll.LoadLibrary("/home/cuidongdong/Pytorch_learning/Reference/exercise9/build/libhello.so")

test.test()
add =test.add
add.argtypes =[c_float, c_float] 
add.restype =c_float
print(add(1.3, 13.4))