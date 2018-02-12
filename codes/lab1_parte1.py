# Procesamiento Digital de Imagenes.
# Laboratorio II.
# Jeckson Jaimes. 12-10446.
# Andres Suarez. 12-10925

import cv2
import matplotlib.pyplot as plt
import numpy as np

# # Parte I - Cargar y mostrar una imagen
imBGR = cv2.imread('../img/con00004.jpg')
imRGB = imBGR[..., ::-1]    # Esto genera el mismo efecto que imgBGR[:,:,::-1]
plt.figure()
plt.title('imRGB')
plt.imshow(imRGB)


plt.figure()
plt.title('HistRGB')
color = ('b', 'g', 'r')

histB = cv2.calcHist([imBGR], [0], None, [256], [0, 256])
histG = cv2.calcHist([imBGR], [1], None, [256], [0, 256])
histR = cv2.calcHist([imBGR], [2], None, [256], [0, 256])
plt.plot(histB,'b')
plt.plot(histG,'g')
plt.plot(histR,'r')
plt.xlim([0, 256])


# Encontrando el maximo y el minimo
# Determinamos el numero maximo de pixeles
px = imBGR.size/3
px_max = px - px*0.05
px_min = px*0.05
maxB = 0
maxR = 0
maxG = 0
minB = 0
minR = 0
minG = 0
px_hist = 0
i = 0
max = 0
min = 0

# Calculo el minimo percentil al 5% de los maximos pixeles
while(px_hist < px_max):
    px_hist += histB[i] 
    i +=1
    if (px_hist < px_max):
        maxB += 1
    if (px_hist < px_min):
        minB += 1

px_hist = 0
i = 0
while(px_hist < px_max):
    px_hist += histR[i] 
    i +=1
    if (px_hist < px_max):
        maxR += 1
    if (px_hist < px_min):
        minR += 1
    
px_hist = 0
i = 0
while(px_hist < px_max):
    px_hist += histG[i] 
    i +=1
    if (px_hist < px_max):
        maxG += 1
    if (px_hist < px_min):
        minG += 1

blue = np.array(imBGR[:,:,0],int)
green = np.array(imBGR[:,:,1],int)
red = np.array(imBGR[:,:,2],int)

consB = 255 / (maxB - minB)
consG = 255 / (maxG - minG)
consR = 255 / (maxR - minR)

imB = (blue - minB) * int(consB)
imG = (green - minG) * int(consG)
imR = (red - minR) * int(consR)

imB = np.array(imB.clip(min=0),np.uint8)
imG = np.array(imG.clip(min=0),np.uint8)
imR = np.array(imR.clip(min=0),np.uint8)


imgST = cv2.merge((imB,imG,imR))
imgST = imgST[..., ::-1]
plt.figure()
plt.title('imRGB sin los pixeles máximos')
plt.imshow(imgST)

plt.figure()
plt.title('HistRGB')
color = ('b', 'g', 'r')

histB = cv2.calcHist([imgST], [0], None, [256], [0, 256])
histG = cv2.calcHist([imgST], [1], None, [256], [0, 256])
histR = cv2.calcHist([imgST], [2], None, [256], [0, 256])
plt.plot(histB,'b')
plt.plot(histG,'g')
plt.plot(histR,'r')
plt.xlim([0, 256])
plt.show()


# Equalizacion

b = cv2.equalizeHist(imBGR[:,:,0]) 
g = cv2.equalizeHist(imBGR[:,:,1])
r = cv2.equalizeHist(imBGR[:,:,2])

imgEQ = cv2.merge((b,g,r))
imgEQ = imgEQ[..., ::-1]
plt.figure() 
plt.title('imRGB Equalizada')
plt.imshow(imgEQ)
plt.show()
