# Procesamiento Digital de Imagenes.
# Laboratorio II.
# Jeckson Jaimes. 12-10446.
# Andres Suarez. 12-10925

import cv2
import matplotlib.pyplot as plt
import numpy as np

def imprimir_img(img,dst):
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]),	plt.yticks([])	
    plt.subplot(122),plt.imshow(dst),plt.title('Promedio')	
    plt.xticks([]),	plt.yticks([])	
    plt.show()

def imprimir_img2x2(img,laplacian,sobelx,sobely):
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplaciano'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()

def filtro_kernel(kn,img):
    kernel = np.ones((kn,kn),np.float)/kn**2
    dst = cv2.filter2D(img,-1,kernel)
    imprimir_img(img,dst)
    	

# # Parte II. Streching Leer imagen de un repositorio de imágenes.

# 1.a) Mascara de suavizado. Kernel 3x3.
# imgBGR = cv2.imread('../img/con00004.jpg')
imgBGR = cv2.imread('../img/Test3.jpg')
imgRGB = imgBGR[..., ::-1]    

filtro_kernel(3,imgRGB)

#  1.b) Kernel 5x5. 

filtro_kernel(5,imgRGB)

# 1.c) Mascara de perfilado.
imgPerfilada = cv2.Laplacian(imgRGB,cv2.CV_8U,ksize=5)
imprimir_img(imgRGB,imgPerfilada)

# 1.d) Difuminado Gaussiano. Elimina el ruido gaussiano de la imagen.
imgBlur = cv2.GaussianBlur(imgRGB,(5,5),0)
imprimir_img(imgRGB,imgBlur)

# 1.e) Detector de contornos. (Imagen en escala de grises)
imgGRAY = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)     # Esta linea se puede sustituir por colocar
                                                        # imgBGR = cv2.imread('../img/Test3.jpg',0) 
                                                        # al leer la imagen.
laplacian = cv2.Laplacian(imgGRAY,cv2.CV_64F)
sobelx = cv2.Sobel(imgGRAY,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(imgGRAY,cv2.CV_64F,0,1,ksize=5)
imprimir_img2x2(imgGRAY,laplacian,sobelx,sobely)

# 1.f) Laplaciano del Gaussiano con Kernel 5x5.
imgGauss = cv2.GaussianBlur(imgRGB,(5,5),0)
imgLaplace = cv2.Laplacian(imgGauss,cv2.CV_8U,ksize=5)
imprimir_img(imgRGB,imgLaplace)

