from PIL import Image
import numpy as np
import struct


a = np.ones((4, 3, 1)).astype(np.float16)
print(a.shape)
print(a.dtype)
print(a)
b = np.ones((3, 4)).astype(np.float32)
#b=np.random.rand(3,3).astype(np.float32)
b[0][0]=0.1
b[0][1]=0.2
b[0][2]=0.3
b[0][3]=0.4
b[1][0]=0.5
b[1][1]=0.6
b[1][2]=0.7
b[1][3]=0.8
b[2][0]=0.9
b[2][1]=1.0
b[2][2]=1.1
b[2][3]=1.2

print(b.shape)
print(b.dtype)
print(b)
print(b.sum())

fname_out = "/home/zwr/EVA_env/eva-02.cpp/temp/template.bin"
with open(fname_out, "wb") as fout:
    a.tofile(fout)
    b.tofile(fout)