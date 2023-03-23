import numpy as np
boxs = np.array([[10, 20, 30, 40],
                 [20, 30, 40, 50],
                 [30, 40, 50, 60]])

xyxy = np.concatenate([boxs[...,:2]-boxs[...,2:]/2,boxs[...,:2]+boxs[...,2:]/2],axis=-1)

# print(xyxy)

a = [[12,34,56,78],[21,43,65,87]]
print(a[1:])
print('====')
print(a[0][2])
print('======')
b = np.array([[12,34,56,78],[21,43,65,87]])
print(b[...,:2])
print('=======')
c = np.array([[[12,34,56,78],[21,43,65,87]]])
print(c[:2])