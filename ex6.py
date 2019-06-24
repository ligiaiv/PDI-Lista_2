from PIL import Image
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import itertools
import time
from scipy import ndimage

sys.setrecursionlimit(10000)


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



def get_neighboors(vector,index, regiao):
	regiao.append(index)

	pixel = vector[:,index]

	vizinhos=np.where(((vector[0,:]>= pixel[0]-1) & (vector[0,:]<= pixel[0]+1))     &     ((vector[1,:]>= pixel[1]-1) & (vector[1,:]<= pixel[1]+1)))
	vizinhos = vizinhos[0]
	for idx in vizinhos:
		if idx not in regiao:
			regiao = get_neighboors(vector,idx,regiao)
	return regiao


start = time.time()

def get_regions(im):
	white = np.vstack(np.where(im == 255))
	regioes = []
	
	while(len(white[0])>0):
		idxs = get_neighboors(white,0,[])
		regiao = white.T[idxs]
		regioes.append(regiao)
		white = np.delete(white,idxs,axis = 1)
	return regioes











im = cv2.imread("imagens/Fig10.40(a).jpg",0).astype(float)
seed_im = np.zeros(im.shape)
seed_im[im>254] =255
white = np.vstack(np.where(seed_im == 255))
mask = np.array([[0,1,0],[1,1,1],[0,1,0]])
cv2.imwrite("imagens/ex6/c - thresholded.png",(seed_im).astype(np.uint8))

regioes = get_regions(seed_im)
print(len(regioes))
# for regiao in regioes:
# 	# print(len)
# 	print(regiao)





seeds = np.ndarray((2,0))
for regiao in regioes:
	seed_im = regiao
	seeds_reg = np.ndarray((2,0))
	while((seed_im==255).sum()>0):
		seed_im_final = seed_im
		seeds_reg = white
		white,seed_im = erosao(seed_im,white,mask)

	seeds = np.vstack([seeds,seeds_reg])
cv2.imwrite("imagens/ex6/d - seeds.png",(seed_im_final).astype(np.uint8))


T = 5
def grow(seed,im,region,T):
	X,Y = seed
	pixel = im[X,Y]
	for x in range(X-1,X+2):
		for y in range(Y-1,Y+2):
			if (x,y) not in region: 
				if np.abs(im[x,y] - pixel) < T :
					region.append((x,y))
					region = grow((x,y),im,region,T) 

	return region



region = []
i = 0
print('shape seeds',seeds.shape)
for seed in seeds.T:
	print("here")
	i+=1
	print(seed)
	new_region = grow(seed,im,region,20)
	region+=new_region
	print(len(region))
	if i==2:
		print(region)
	# 	quit()
print("out")



region = np.array(region).T
print("here2")
print(region.shape)
new_im = np.zeros(im.shape)
print("here3")

new_im[region[0],region[1]] = 255
print("here4")

cv2.imwrite("imagens/ex6/i - region.png",(new_im).astype(np.uint8))
print("here5")

# white = np.vstack(np.where(seed == 255))
# black = np.vstack(np.where(seed == 0))

end = time.time()

print('Tempo de Execução: ', end-start)