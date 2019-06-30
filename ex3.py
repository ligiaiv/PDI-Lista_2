import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import itertools
import time,os

if not os.path.exists("imagens"):
	print("Crie uma pasta 'imagens/' no mesmo diretorio deste script. As imagens originais devem estar nesta pasta.")
	quit()
elif not os.path.exists("imagens/ex3"):
    os.makedirs("imagens/ex3")


def pad(im,B):
	l = B.shape[0]//2
	h = B.shape[1]//2
	padded = np.zeros((im.shape[0]+2*l,im.shape[1]+2*h))
	padded[l:im.shape[0]+l,h:im.shape[1]+h] = im
	return padded
def unpad(im,B):
	l = B.shape[0]//2
	h = B.shape[1]//2
	unpad = im[l:-l,h:-h]
	return unpad

def erosao(im,A,B):
	new_A = np.ndarray(shape = (2,0),dtype = int)
	padded = pad(im,B)
	l = B.shape[0]//2
	h = B.shape[1]//2
	for pixel in A.T:
		x = pixel[0]+l
		y = pixel[1]+h


		partial = padded[x-l:x+l+1,y-h:y+h+1]/255
		if (bool(partial[B.astype(bool)].min())):
			new_A = np.hstack([new_A,pixel[:,None]])

	erosado = np.zeros(im.shape)

	erosado[(new_A[0],new_A[1])] = 255
	return new_A,erosado

#im = padded image
#A = original object, no padding
def dilatacao(im,A,B):


	new_A = A.copy()
	padded = pad(im,B)
	l = B.shape[0]//2
	h = B.shape[1]//2
	for pixel in A.T:
		x = pixel[0]+l
		y = pixel[1]+h
		# print('pixel',pixel)

		partial = padded[x-l:x+l+1,y-h:y+h+1]/255
		if (bool(partial[B.astype(bool)].max())):

			new_A = new_A[:,np.invert(np.all(new_A.T==pixel,axis = 1)).T]

	dilatado = np.ones(im.shape)*255
	dilatado[(new_A[0],new_A[1])] = 0

	return new_A,dilatado

def abertura(im,B):
	white = np.vstack(np.where(im == 255))
	_,im2 = erosao(im,white,B)
	black = np.vstack(np.where(im2 == 0))
	return dilatacao(im2,black,B)


start = time.time()

im = cv2.imread("imagens/Fig11.10.jpg",0).astype(float)
im[im>100] =255
im[im<=100] = 0
white = np.vstack(np.where(im == 255))
black = np.vstack(np.where(im == 0))

mask = np.array([[0,1,0],[1,1,1],[0,1,0]])

novo = im
K_max = 0
while (novo == 255).sum() >0:
	K_max+=1
	velho = novo
	white = np.vstack(np.where(velho == 255))
	_,novo = erosao(velho,white,mask) 
	print((novo==255).sum())
print('K_max: ',K_max)
total = np.zeros(im.shape).astype(int)
for K in range(0,K_max):
	novo = im
	print("- - - k = ",K)
	for k in range(1,K+1):
		print(k,'/',K) 
		velho = novo
		white = np.vstack(np.where(velho == 255))
		_,novo = erosao(velho,white,mask) 
	_,aberto = abertura(novo,mask)

	SkA = novo - aberto
	SkA[SkA<0]=0
	total = np.bitwise_or(total,(SkA/255).astype(int))
end = time.time()
cv2.imwrite("imagens/ex3/esqueleto.jpg",(total*255).astype(np.uint8))

print('Tempo de Execução: ', end-start)