from PIL import Image
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import itertools
im = cv2.imread("imagens/Thyroid.jpg",1)


I = im.sum(axis=2)/3

minimo = im.min()
maximo = im.max()
n_slices = 8
slice_size = int((maximo-minimo)/n_slices)

new_im = np.zeros(shape = (im.shape[0],im.shape[1],3))

# color = np.zeros(shape=(3), dtype=np.uint8)

colors = list(itertools.product(range(2), repeat=3))
for x in range(n_slices):
	bottom = x*slice_size + minimo
	top = bottom + slice_size
	color = np.asarray(colors[x])*255
	new_im[np.bitwise_and(I>=bottom,I<top)] = color

cv2.imwrite("imagens/ex1/sliced.jpg",new_im)
