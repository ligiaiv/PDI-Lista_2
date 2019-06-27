from PIL import Image
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import itertools
import time
from scipy import ndimage

def get_mpq(p,q,image):
	m,n = image.shape
	x_vector = (np.arange(m)**p)[:,None]
	y_vector = (np.arange(n)**q)[None,:]
	mpq = ((x_vector*y_vector)*image).sum()
	return mpq

def get_mu_pq(p,q,image):
	m00 = image.sum()
	x_ = get_mpq(1,0,image)/m00
	y_ = get_mpq(0,1,image)/m00

	m,n = image.shape
	x_vector = (np.arange(m)**p)[:,None] - x_
	y_vector = (np.arange(n)**q)[None,:] - y_
	mu_pq = ((x_vector*y_vector)*image).sum()
	return mu_pq

def get_eta_pq(p,q,image):

	gama = (p+q)/2 + 1

	m,n = image.shape
	mu_00 = image.sum()
	eta_pq = get_mu_pq(p,q,image)/mu_00
	return eta_pq

def moment_invariants(image):
	eta20 = get_eta_pq(2,0,image)
	eta02 = get_eta_pq(0,2,image)
	eta11 = get_eta_pq(1,1,image)
	eta12 = get_eta_pq(1,2,image)
	eta21 = get_eta_pq(2,1,image)
	eta03 = get_eta_pq(0,3,image)
	eta30 = get_eta_pq(3,0,image)

	phi1 = eta20+eta02
	phi2 = (eta20-eta02)**2 + 4*eta11**2
	phi3 = (eta30-3*eta12)**2 + (3*eta21-eta03)**2
	phi4 = (eta30 + eta12)**2 + (eta21 +eta03)**2
	phi5 = (eta30 - 3*eta12)*((eta30+eta12)**2 - 3*(eta21+eta03)**2)+
			(3*eta21-eta03)*(eta21+eta03)*(3*(eta30+eta12)**2 -(eta21+eta03)**2)
	phi6 = (eta20-eta02)*((eta30+eta12)**2-(eta21+eta03)**2)+4*eta11*(eta30+eta12)*(eta21+eta03)
	phi7 = (3*eta21 - eta03)*(eta30+eta12)*((eta30+eta12)**2-3*(eta21+eta03)**2)+
			(3*eta12 - eta30)*(eta21+eta03)*(3*(eta30+eta12)**2 - (eta21+eta03)**2)

	return [phi1,phi2,phi3,phi4,phi5,phi6,phi7]

		










im = cv2.imread("imagens/lena.tif",0).astype(float)
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
	seeds = np.hstack([seeds,seeds_reg])
	print('Seeds.shape',seeds.shape)

cv2.imwrite("imagens/ex6/d - seeds.png",(seed_im_final).astype(np.uint8))
print('Seeds.shape',seeds.shape)

T = 5

def grow(seed,im,region,T,i):
	X,Y = seed
	# print('X',X,'Y',Y)
	i+=1
	# print('i:',i)
	pixel = im[X,Y]
	for x in range(np.max([X-1,0]),np.min([X+2,im.shape[0]])):
		for y in range(np.max([Y-1,0]),np.min([Y+2,im.shape[1]])):
			if (x,y) not in region: 
				if np.abs(im[x,y] - pixel) < T :
					# print("diferenca:",np.abs(im[x,y] - pixel))
					# print("append")
					region.append((x,y))
					region = grow((x,y),im,region,T,i) 

	return region



region = []
i = 0
print('Seeds final',seeds.shape)
seeds = seeds.astype(int)
for seed in seeds.T:
	# print("here")
	# print(seed)

	region = grow(seed,im,region,5,0)
	# region+=new_region
	# print(len(region))
	# if i==2:
	# 	print(region)
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