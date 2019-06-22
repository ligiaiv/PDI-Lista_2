from PIL import Image
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import itertools
import time
from scipy import ndimage

# partials =  [bool([im[x-l:x+l+1,y-h:y+h+1]/255][B.astype(bool)].max()) for x in range(1,im.shape[0]-1) for y in range(1,im.shape[1]-1)]
# quit()
# def get_neighboor(im,B):
# 	m = B.shape[0]//2
# 	n = B.shape[0]//2
# 	new_array = np.zeros(im.shape)
# 	padded = np.np.zeros((im.shape[0]+m//2,im.shape[1]+n//2))[m==]
# 	for x in range(m,im.shape[0]-m):
# 		for y in range(n,im.shape[0]-n):

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
	# print('A',A.shape)
	# print('B',B.shape)
	# print('im',im.shape)
	# print(A)

	# new_A = np.ndarray(shape = (2,0),dtype = int)
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
			# print('.',np.invert(np.all(new_A.T==pixel,axis = 1)).sum())
			# print(np.invert(np.all(new_A.T==pixel,axis = 1)).shape)
			# print(new_A.shape)
			new_A = new_A[:,np.invert(np.all(new_A.T==pixel,axis = 1)).T]
			# print('.',np.where(np.bitwise_and(new_A[0]==x,new_A[1==y])))
			# new_A = np.delete(new_A, np.argwhere(new_A.T==pixel))
			# new_A = np.hstack([new_A,pixel[:,None]])
	# new_white = np.concatenate((new_A, white), axis=1)
	dilatado = np.ones(im.shape)*255
	# print(new_A.shape)
	dilatado[(new_A[0],new_A[1])] = 0

	return new_A,dilatado

def abertura(im,B):
	white = np.vstack(np.where(im == 255))
	_,im2 = erosao(im,white,B)
	# print('im2:',im2)
	black = np.vstack(np.where(im2 == 0))
	# print(im2.shape)
	return dilatacao(im2,black,B)

# def dilatacao(im,B):
# 	new_A = np.ndarray(shape = (2,0),dtype = int)
# 	l = B.shape[0]//2
# 	h = B.shape[1]//2
# 	padded = np.zeros((im.shape[0]+2*l,im.shape[1]+2*h))
# 	padded[l:im.shape[0]+l,h:im.shape[1]+h] = im

# 	new_im = np.ndarray(shape = im.shape)
# 	for x_ in range(im.shape[0]-1):
# 		x = x_+l
# 		for y_ in range(im.shape[1]-1):
# 			y = y_+l

# 			partial = padded[x-l:x+l+1,y-h:y+h+1]/255
# 			# print(x,y)
# 			new_im[x,y] = bool(partial[B.astype(bool)].max())
# 		# if (bool(partial[B.astype(bool)].max())):
# 		# 	new_A = np.hstack([new_A,pixel[:,None]])

# 	return new_im

start = time.time()

im = cv2.imread("imagens/Fig11.10.jpg",0).astype(float)
im[im>100] =255
im[im<=100] = 0
white = np.vstack(np.where(im == 255))
black = np.vstack(np.where(im == 0))

mask = np.array([[0,1,0],[1,1,1],[0,1,0]])
# mask = np.ones((3,3))
# im = np.array([[1,0,0,0,0],[0,1,1,0,0],[0,1,1,0,0],[0,1,1,1,0],[0,1,1,1,0]])
# im = np.vstack([im,np.ones((5,5))])
# im = im*255
# cv2.imwrite("imagens/ex3/im_livro.jpg",im.astype(np.uint8))

# padded = pad(im,mask)

# new_white = np.concatenate((dilatacao(padded,black,mask), white), axis=1)
# new_im2 = ndimage.binary_erosion(im/255, structure = erosao_mask)
# print('velho ',white.shape,' novo ',new_white.shape, 'lib', (new_im2==1).sum() )

# cv2.imwrite("imagens/ex3/erosao.jpg",new_im.astype(np.uint8))
# cv2.imwrite("imagens/ex3/dilatado.jpg",dilatado.astype(np.uint8)*255)
# dilatado = np.zeros(im.shape)

# dilatado[(new_white[0],new_white[1])] = 255



# _,aberto = abertura(im,mask) 
# aberto_lib = ndimage.binary_opening(im,structure = mask)
# print('ABERTURA meu ',(aberto==0).astype(int).sum(),' lib ',(aberto_lib==0).astype(int).sum())

# _,dilatado = dilatacao(im,black,mask) 
# dilatado_lib = ndimage.binary_dilation(im,structure = mask)
# print('DILATAÇÃO meu ',(dilatado==0).astype(int).sum(),' lib ',(dilatado_lib==0).astype(int).sum())


# _,erosado = erosao(im,white,mask) 
# erosado_lib = ndimage.binary_erosion(im,structure = mask)
# print('EROSAO meu ',(erosado==0).astype(int).sum(),' lib ',(erosado_lib==0).astype(int).sum())



# _,aberto = abertura(im,mask)
# cv2.imwrite("imagens/ex3/dilata.jpg",aberto.astype(np.uint8))
# cv2.imwrite("imagens/ex3/aberto.jpg",aberto.astype(np.uint8))
# teste1 = np.array([[0,1,0],[0,1,0],[1,1,1],[1,1,1],[1,1,1]])*255
# print(teste1)
# _,teste1_aberto = abertura(teste1,mask)
# cv2.imwrite("imagens/ex3/aberto_teste1.jpg",(teste1_aberto*255).astype(np.uint8))
# print('aberto_teste1',teste1_aberto)
# quit()
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
	# cv2.imwrite("imagens/ex3/novo_"+str(K)+".jpg",(novo).astype(np.uint8))
	# cv2.imwrite("imagens/ex3/aberto_"+str(K)+".jpg",(aberto).astype(np.uint8))

	SkA = novo - aberto
	SkA[SkA<0]=0
	total = np.bitwise_or(total,(SkA/255).astype(int))
end = time.time()
cv2.imwrite("imagens/ex3/esqueleto.jpg",(total*255).astype(np.uint8))
# cv2.imwrite("imagens/ex3/novo.jpg",(novo).astype(np.uint8))

print('Tempo de Execução: ', end-start)