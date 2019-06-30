import numpy as np
def matrix_(im):
	im_3d = np.ndarray(shape = im.shape+(0,),dtype = int)
	# print(im_3d.shape)
	for delta_x in range(-1,2):
		for delta_y in range(-1,2):
			shifted = np.roll(im, delta_x, axis=0)
			shifted = np.roll(shifted, delta_y, axis=1)[:,:,None]
			# print('s',shifted)
			# print(shifted.shape)
			im_3d = np.concatenate((im_3d,shifted), axis=2)
			# print(im_3d)
			# im_3d = np.dstack((im_3d,shifted))
	print(im_3d.shape)
	return im_3d
im = np.arange(3)[:,None] * np.arange(3)[None,:]

black = np.vstack(np.where(seed == 0))

for (x,y) in black:
	pixel = im[x,y]
	partial = ###
	if ((np.abs(partial - pixel) < T).sum() > 0)
		white.remove()


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
print(im)
im_3d = matrix_(im)
im_3d[np.abs(im-im_3d) < T]