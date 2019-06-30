import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import itertools
import time,os

from conv2D import conv2D


if not os.path.exists("imagens"):
	print("Crie uma pasta 'imagens/' no mesmo diretorio deste script. As imagens originais devem estar nesta pasta.")
	quit()
elif not os.path.exists("imagens/ex5"):
    os.makedirs("imagens/ex5")


start = time.time()




# Lê a imagem colorida
im = cv2.imread("imagens/estrada.png",1).astype(float)

# Calcula o grayscale e faz o limiar para encontrar as linhas brancas
im_gray = im.sum(axis = 2)/3
cv2.imwrite("imagens/ex5/estrada_gray.png",im_gray.astype(np.uint8))
im_gray[im_gray<230] = 0
im_gray[im_gray>=230] = 255

cv2.imwrite("imagens/ex5/estrada_gray_limiar.png",im_gray.astype(np.uint8))

# Tenta encontrar as linhas brancas na imagem colorida tentando encontrar os pixels cujas 3 cores possuam alto valor
im_cor = np.zeros(im_gray.shape)
im_cor[np.invert(np.bitwise_and(im[:,:,0]<250 , im[:,:,1]<250 , im[:,:,2]<250))]=255
cv2.imwrite("imagens/ex5/estrada_cor_or.png",(np.invert(im_cor.astype(bool))*255).astype(np.uint8))

im_cor = np.zeros(im_gray.shape)
im_cor[np.invert(np.bitwise_or(im[:,:,0]<250 , im[:,:,1]<250 , im[:,:,2]<250))]=255
cv2.imwrite("imagens/ex5/estrada_cor_and.png",(np.invert(im_cor.astype(bool))*255).astype(np.uint8))

# Une os resultados das 2 técnicas para uma imagem melhor
im_gray = np.bitwise_and(im_cor.astype(bool),im_gray.astype(bool))*255
cv2.imwrite("imagens/ex5/estrada_mix.png",(np.invert(im_gray.astype(bool))*255).astype(np.uint8))
im_gray = im_cor

# Filtro de sobel que foi implementado, porém não foi usado pois outra técnica produziu resultado melhor
def sobel(im):
	sobel_x = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
	sobel_y = np.flip(sobel_x,axis = 0)
	print(sobel_y)
	conv = conv2D()
	gx = conv.make_conv(im,sobel_x,'zeroes')
	gy = conv.make_conv(im,sobel_y,'zeroes')
	cv2.imwrite("imagens/ex5/gx.png",gx.astype(np.uint8))
	cv2.imwrite("imagens/ex5/gy.png",gy.astype(np.uint8))


	result = np.abs(gx)+np.abs(gy)
	cv2.imwrite("imagens/ex5/(gx+gy)4.png",(result/4).astype(np.uint8))

	result[result<520] = 0
	result[result>=520] = 255
	cv2.imwrite("imagens/ex5/result.png",(result).astype(np.uint8))

	print((result>255).sum())
# sobel(im_gray)



# Função da transformada de Hough
def hough(im):
	tetas = np.deg2rad(np.arange(-90, 90,0.5)) #cria vetor de valores de -90 a 90 graus e converte para radianos
	width, height = im.shape
	diag_len = np.ceil(np.sqrt(width**2 + height**2))
	rhos = np.arange(-diag_len, diag_len) #cria vetor de rhos

	plano_parametros = np.zeros((len(rhos),len(tetas)),dtype = np.uint64) #cria a matriz que será usada como base para desenhar o plano de parâmetros(rho x teta)
	y_idxs, x_idxs = np.nonzero(im) #pega índices de pixeis selecionados anteriormente (brancos)

	# Para cada ponto x,y branco na figura limiarizada, cria uma função pho(teta) com parametros x,y
	# "plota" essa função no plano de parâmetros
	for i in range(len(x_idxs)):
		x = x_idxs[i]
		y = y_idxs[i]
		rho = np.round(x * np.cos(tetas) + y * np.sin(tetas)) + diag_len

		plano_parametros[rho.astype(np.uint64), np.arange(len(tetas))] += 1

		#imprime a primeira função "plotada" só para debug e relatório
		if(i==0):
			cv2.imwrite("imagens/ex5/reta_1.png",np.flip((plano_parametros*255),axis = 0).astype(np.uint8))

	return plano_parametros, tetas, rhos


plano_parametros, tetas, rhos = hough(im_gray)

#imprime o plano de parâmetros invertendo o eixo vertical, pois a direção na imagem é inversa ao do plano cartesiano
plano_parametros_img = np.flip(plano_parametros*255/np.max(plano_parametros),axis = 0) 
print("MAX",np.max(plano_parametros))
cv2.imwrite("imagens/ex5/plano_parametros.png",(plano_parametros_img*25).astype(np.uint8))

#pega os 4 pixels de maior luminosidade. Obs: 4 pois devido a a questões de resolução o maior ponto de luminosidade ocupava mais de 1 pixel
N = 4
idxs = np.argsort(plano_parametros.ravel())[-N:] 

#pega os 2 primeiros do vetor de 4. Talvez por coincidência os 2 de luminosidade pontos separados apareciam no início do vetor, mas apenas de anteriormente fossem pegos 4
#essa parte foi feita a partir de observação dos resultados e ajuste do código. Provavelmente deve ser melhor ajustada para poder trabalhar com outras imagens
rho = rhos[idxs // plano_parametros.shape[1]][:2]
teta = tetas[idxs % plano_parametros.shape[1]][:2]

#recria as retas no plano das variáveis (x,y)
x = np.arange(-10,10,0.1)
y = (rho - x[:,None]*np.cos(teta))/np.sin(teta)

print("rho: ", rho, " teta: ",np.rad2deg(teta))

fig = plt.figure()

plt.subplot(1, 2, 2)
plt.plot(x, y[:,0])
plt.subplot(1, 2, 1)

plt.plot(x, y[:,1])

plt.show()
end = time.time()

print('Tempo de Execução: ', end-start)