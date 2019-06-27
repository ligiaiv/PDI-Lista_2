import numpy as np
import cv2
import pandas as pd
import time

def get_mpq(p,q,image):
	m,n = image.shape
	x_vector = (np.arange(m)**p)[:,None]
	y_vector = (np.arange(n)**q)[None,:]
	# print((x_vector*y_vector)*image)
	mpq = ((x_vector*y_vector)*image).sum()
	return mpq

def get_mu_pq(p,q,image):
	m00 = image.sum()
	# m00 = get_mpq(0,0,image)
	x_ = get_mpq(1,0,image)/m00
	y_ = get_mpq(0,1,image)/m00

	m,n = image.shape
	x_vector = ((np.arange(m) - x_)**p)[:,None]
	y_vector = ((np.arange(n) - y_)**q)[None,:]
	mu_pq = ((x_vector*y_vector)*image).sum()
	return mu_pq

def get_eta_pq(p,q,image):

	gama = ((p+q)/2) + 1

	m,n = image.shape
	mu_00 = image.sum()
	# mu_00 = get_mu_pq(0,0,image)
	eta_pq = get_mu_pq(p,q,image)/np.power(mu_00,gama)
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
	phi2 = (eta20-eta02)**2 + 4*(eta11**2)
	phi3 = (eta30-3*eta12)**2 + (3*eta21-eta03)**2
	phi4 = (eta30 + eta12)**2 + (eta21 +eta03)**2
	phi5 = (eta30 - 3*eta12)*((eta30+eta12)**2 - 3*(eta21+eta03)**2)+\
			(3*eta21-eta03)*(eta21+eta03)*(3*(eta30+eta12)**2 -(eta21+eta03)**2)
	phi6 = (eta20-eta02)*((eta30+eta12)**2-(eta21+eta03)**2)+4*eta11*(eta30+eta12)*(eta21+eta03)
	phi7 = (3*eta21 - eta03)*(eta30+eta12)*((eta30+eta12)**2-3*(eta21+eta03)**2)+\
			(3*eta12 - eta30)*(eta21+eta03)*(3*(eta30+eta12)**2 - (eta21+eta03)**2)

	phi = np.array([phi1,phi2,phi3,phi4,phi5,phi6,phi7])
	phi = np.sign(phi)*np.abs(np.log10(np.abs(phi))).tolist()
	return phi







# imex = np.ones((10,10))*5
# print(get_mu_pq(0,0,im))
# print(im.sum())

# quit()




imagens = {}
im = cv2.imread("imagens/lena.tif",0).astype(float)

rows,cols = im.shape
imagens['im'] = im
imagens['reduzido'] = cv2.resize(im,None,fx=0.5, fy=0.5)

M90 = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
M180 = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),180,1)

imagens["90"] = cv2.warpAffine(im,M90,(cols,rows))
imagens["80"] = cv2.warpAffine(im,M180,im.shape)

start = time.time()
moment_dict = {}
for name,img in imagens.items():
	cv2.imwrite("imagens/ex8/"+name+".png",img.astype(np.uint8))
	moment_dict[name]=moment_invariants(img)
df = pd.DataFrame.from_dict(moment_dict, orient='index',columns=list(range(1,8)))
print(df)
# cv2.imwrite("imagens/ex8/reduzida.png",reduzido.astype(np.uint8))
# cv2.imwrite("imagens/ex8/rot90.png",rotacionado90.astype(np.uint8))
# cv2.imwrite("imagens/ex8/rot180.png",rotacionado180.astype(np.uint8))
end = time.time()

print('Tempo de Execução: ', end-start)