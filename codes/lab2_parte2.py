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

def filtro_kernel(kn,img):
    kernel = np.ones((kn,kn),np.float32)/kn**2
    dst = cv2.filter2D(img,-1,kernel)
    imprimir_img(img,dst)
    	

# # Parte II 

# 1.a) Mascara de suavizado
# imgBGR = cv2.imread('../img/con00004.jpg')
imgBGR = cv2.imread('../img/Test3.jpg')
imgRGB = imgBGR[..., ::-1]    

# plt.figure()
# plt.title('imRGB')
# plt.imshow(imgRGB)

# filtro_kernel(3,imgRGB)

# # 1.b)
# filtro_kernel(5,imgRGB)

# 1.c)
imgPerfilada = cv2.Laplacian(imgRGB,cv2.CV_8U,ksize=1)
imprimir_img(imgRGB,imgPerfilada)

# 1.d)

