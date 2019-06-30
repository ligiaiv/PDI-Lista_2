from PIL import Image
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import itertools
import time
from scipy import ndimage

sys.setrecursionlimit(20000)


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
print("REGIOES:",len(regioes))
# for regiao in regioes:
# 	# print(len)
# 	print(regiao)




seeds = np.ndarray((2,0))
i  =0
for regiao in regioes:
	regiao = np.array(regiao).T
	print('regiao',regiao.shape)



	i+=1
	im_reg = np.zeros(im.shape)
	# print("maxim",regiao[1].max())
	im_reg[regiao[0],regiao[1]] = 255
	
	white_reg = regiao
	print("\n\n-----------------------------",i)
	seeds_reg = np.ndarray((2,0))
	while((im_reg==255).sum()>0):
		seed_im_final = seed_im
		seeds_reg = white_reg
		white_reg,im_reg = erosao(im_reg,white_reg,mask)
		print('white_reg',white_reg.shape)
	print('seeds',seeds.shape)
	print('seeds_reg',seeds_reg.shape)
	random_seed = seeds_reg[:,np.random.randint(0,seeds_reg.shape[1])][:,None]
	seeds = np.hstack([seeds,random_seed])
	print('Seeds.shape',seeds.shape)
seed_im_final = np.zeros(im.shape)
print("SEEDS",seeds)
seeds = seeds.astype(int)

seed_im_final[seeds[0,:],seeds[1,:]] = 255
cv2.imwrite("imagens/ex6/d - seeds.png",(seed_im_final).astype(np.uint8))
print('Seeds.shape',seeds.shape)

T = 10
im_3d = np.ndarray(shape = im.shape+(0,),dtype = int)

for delta_x in range(-1,1):
	for delta_y in range(-1,1):
		shifted = np.roll(im, delta_x, axis=0)
		shifted = np.roll(shifted, delta_y, axis=0)
		im_3d = np.dstack((im_3d,shifted))
def grow(seed,im,region,T,i):
	X,Y = seed
	# print('X',X,'Y',Y)
	i+=1
	print('i:',i)
	pixel = im[X,Y]
	for x in range(np.max([X-1,0]),np.min([X+2,im.shape[0]])):
		for y in range(np.max([Y-1,0]),np.min([Y+2,im.shape[1]])):
			if (x,y) not in region: 
				if np.abs(im[x,y] - pixel) < T :
					# print("diferenca:",np.abs(im[x,y] - pixel))
					# print("append")
					region.append((x,y))
					region = grow((x,y),im,region,T,i) 
				else:
					print("bigger")
			# else:
				# print("in")
	return region

def grow2(im,region,new_pixels,T):
	new_region = region.copy()
	white = np.vstack(np.where(new_pixels == 1))
	for x,y in white.T:
		pixel = im[x,y]
		partial = im[x-1:x+2,y-1:y+2]
		ok = (np.abs(partial - pixel) < T)
		new_region[x-1:x+2,y-1:y+2] = np.bitwise_or(new_region[x-1:x+2,y-1:y+2],ok)
	new_pixels = new_region^region
	return new_region,new_pixels

region = []
i = 0
print('Seeds final',seeds.shape)
seeds = seeds.astype(int)

total_branco = []
for T in range(1,6):
	region2 = seed_im_final.astype(bool)
	region1 = np.zeros(region2.shape)
	print('\nr1',region1.sum())
	print('r2',region2.sum())
	new = region2.copy()
	while new.sum()>0:
		# print("oi")
		region1 = region2.copy()
		region2,new = grow2(im,region2,new,T)
		print('\nnew',new.sum())
		print('r2',region2.sum())
	total_branco.append(region2.sum())
	cv2.imwrite("imagens/ex6/i - region - T"+str(T)+".png",(region2*255).astype(np.uint8))
print(total_branco)
# for seed in seeds.T:
# 	# print("here")
# 	# print(seed)

# 	region = grow(seed,im,region,T,0)
# 	# region+=new_region
# 	# print(len(region))
# 	# if i==2:
# 	# 	print(region)
# 	# 	quit()
# print("out")



# region = np.array(region).T
# print("here2")
# print(region.shape)
# new_im = np.zeros(im.shape)
# print("here3")

# new_im[region[0],region[1]] = 255
# print("here4")

print("here5")

# white = np.vstack(np.where(seed == 255))
# black = np.vstack(np.where(seed == 0))

end = time.time()

print('Tempo de Execução: ', end-start)