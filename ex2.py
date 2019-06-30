from PIL import Image
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import itertools


def BGR2HSI(im):
	R = im[:,:,2]
	G = im[:,:,1]
	B = im[:,:,0]
	I = im.sum(axis=2)/3
	cv2.imwrite("imagens/ex2/I.jpg",I.astype(np.uint8))

	has_color = np.invert(np.bitwise_and(R == B, R == G))
	gray = (np.bitwise_and(R == B, R == G))

	teta = np.ndarray(shape =R.shape)
	# raiz = np.sqrt(np.square(R-G) + ((R-B)*(G-B)))[has_color]
	# print('raiz',(raiz<10).astype(int).sum(),' min',raiz.min(),' max', raiz.max())
	# cima = ((((R-G)+(R-B))/2)[has_color])
	# print('cima',(cima>1).astype(int).sum(),' min',cima.min(),' max', cima.max())

	middle = ((((R-G)+(R-B))/2)[has_color])/(np.sqrt(np.square(R-G) + ((R-B)*(G-B)))[has_color])
	# print(middle.shape)
	# print('middle',(middle>1).astype(int).sum()/np.ones(middle.shape).sum(),' max', middle.max())
	teta[has_color] = np.arccos(middle)
	teta[gray] = np.pi/2
	
	H = np.ndarray(shape =R.shape)
	S = np.ndarray(shape =R.shape)

	H[B<=G] = teta[B<=G]
	H[B>G] = 2*np.pi-teta[B>G]
	H[I==0] = 0
	S[I==0] = 0
	print('nan: ',np.isnan(H).astype(int).sum())
	S[I!=0] = 1-(1/I[I!=0])*(np.amin(im,axis = 2)[I!=0])
	return(H,S,I)
def HSI2BGR(H,S,I):
	R = np.ndarray(H.shape)

	G = np.ndarray(H.shape)
	B = np.ndarray(H.shape)
	cores = [B,R,G]

	for x in range(3):
		low = x*2*np.pi/3
		high = low+2*np.pi/3
		Hx = H-x*2*np.pi/3
		cond = np.bitwise_and(H>=low,H<high)
		cores[(x+0)%3][cond] = I[cond]*(1-S[cond])

		cores[(x+1)%3][cond] = I[cond]*(1 + S[cond]*np.cos(Hx[cond])/np.cos((np.pi/3)-Hx[cond]))
		cores[(x+2)%3][cond] = 3*I[cond] - (cores[(x+0)%3][cond]+cores[(x+1)%3][cond])
	im = np.stack((B, G,R), axis=2)
	return im


im = cv2.imread("imagens/peppers.tiff",1).astype(np.float64)
# print(type(im)) 
# im[:,:,2] = 1-im[:,:,2]
R = im[:,:,2]
G = im[:,:,1]
B = im[:,:,0]
cv2.imwrite("imagens/ex2/R.jpg",R)
nR = np.zeros(R.shape)
nB = np.zeros(R.shape)
aj = 200
W = 20

nR[R>180] = 255
# nR[np.absolute(R-aj)<W/2] = 255
H,S,I = BGR2HSI(im)
ajuste = 0
ajuste2 = 0.2
low = (2*np.pi - np.pi/6) + ajuste
high = np.pi/6 + ajuste
H[np.bitwise_or(H<high,H>low)] =(H[np.bitwise_or(H<high,H>low)]+np.pi+ajuste2)%(2*np.pi) 
im2 = HSI2BGR(H,S,I)
cv2.imwrite("imagens/ex2/nR.jpg",nR)
cv2.imwrite("imagens/ex2/im2.jpg",im2)
