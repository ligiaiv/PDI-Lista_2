import numpy as np

class conv2D:
	# faz convolução em 1 pixel
	def multiplicar_pixel(self,pixel,matrix,mask):
		x = pixel[0]
		y = pixel[1]
		N2 = mask.shape[0]//2
		partial = matrix[x-N2:x+N2+1,y-N2:y+N2+1]

		result = (partial*mask).sum()
		return result
	# ajusta o padding e faz convolução
	def mediana_pixel(self,pixel,matrix,mask):
		x = pixel[0]
		y = pixel[1]
		N2 = mask.shape[0]//2
		partial = matrix[x-N2:x+N2+1,y-N2:y+N2+1]

		result = np.median(partial)
		return result
	def mediana_adaptativa_pixel(self,pixel,matrix):
		# print("here")
		x = pixel[0]
		y = pixel[1]
		N2 = 1
		Smax = 11//2
		while N2<Smax:

			partial = matrix[x-N2:x+N2+1,y-N2:y+N2+1]
			zmin = np.min(partial)
			zmax = np.max(partial)
			zmed = np.median(partial)
			zxy = matrix[x,y]

			A1 =zmed-zmin
			A2 = zmed-zmax
			if A1>0 and A2<0:
				B1 = zxy-zmin
				B2 = zxy-zmax
				if B1>0 and B2<0:
					result = zxy
				else:
					result = zmed
				break
			else:
				N2 = N2+1
				result = zmed
		return result
	# ajusta o padding e faz convolução
	def make_conv(self,matrix,mask,option,funct = "mult"):
		print(mask)
		N2 = mask.shape[0]//2

		if option == 'reduce':
			matrix = matrix

		elif option == 'zeros':
			matrix = np.hstack((np.zeros((matrix.shape[0],N2)),matrix,np.zeros((matrix.shape[0],N2))))
			matrix = np.vstack((np.zeros((N2,matrix.shape[1])),matrix,np.zeros((N2,matrix.shape[1]))))

		elif option == 'ones':
			matrix = np.hstack((np.ones((matrix.shape[0],N2))*255,matrix,np.ones((matrix.shape[0],N2))*255))
			matrix = np.vstack((np.ones((N2,matrix.shape[1]))*255,matrix,np.ones((N2,matrix.shape[1]))*255))

		# elif option == 'copy':
		# 	matrix = np.hstack(np.flip(matrix,axis = 1)[:,-1*N2:],matrix,np.flip(matrix,axis = 1)[:,:N2]) 
		# 	matrix = np.vstack(np.flip(matrix,axis = 1)[:,-1*N2:],matrix,np.flip(matrix,axis = 1)[:,:N2]) 


		new_shape_x = matrix.shape[0]-mask.shape[0]+1
		new_shape_y = matrix.shape[1]-mask.shape[1]+1

		new_matrix = np.ndarray((new_shape_x,new_shape_y))

		if funct == "mult":
			for x in range(new_shape_x):
				for y in range(new_shape_y):
					new_matrix[x,y] = self.multiplicar_pixel((x+N2,y+N2),matrix,mask)
		elif funct == "mediana":
			for x in range(new_shape_x):
				for y in range(new_shape_y):
					new_matrix[x,y] = self.mediana_pixel((x+N2,y+N2),matrix,mask)
		elif funct == "mediana_adaptativa":
			for x in range(new_shape_x):
				for y in range(new_shape_y):
					new_matrix[x,y] = self.mediana_adaptativa_pixel((x+N2,y+N2),matrix)
		return new_matrix

