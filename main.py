import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageGrab
from scipy.interpolate import lagrange

dic = r'img.png'
img = cv.imread(dic)

cor_para_amarelo = (255, 0, 0)

im = Image.open(dic).convert('RGB')

data = np.array(im)

vermelho, verde, azul = data.T

condicao = (vermelho >= 200) & (verde >= 150) & (azul <= 50)
data[condicao.T] = cor_para_amarelo

im2 = Image.fromarray(data)
im2.save("seeYellow.png")

img = cv.imread("seeYellow.png")

grey = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

kernel = np.ones((2, 2), np.uint8)
grey = cv.GaussianBlur(grey, (9, 9), 0)
grey = cv.morphologyEx(grey, cv.MORPH_OPEN, kernel)
grey = cv.morphologyEx(grey, cv.MORPH_CLOSE, kernel)

canny = cv.Canny(grey, 170, 200)

circles = cv.HoughCircles(canny,
                          cv.HOUGH_GRADIENT,
                          dp=1.1,
                          minDist=26,
                          param1=27,
                          param2=7,
                          minRadius=0,
                          maxRadius=9)

# Changing the dtype  to int
circles = np.uint16(np.around(circles))
cimg = canny.copy()
for i in circles[0, :]:
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (255, 0, 0), 10)


red_image = PIL.Image.open(dic)
red_image_rgb = red_image.convert("RGB")
counter = 0
verde = 0
vermelho = 0
amarelo = 0

for i in circles[0, :]:
    rgb_pixel_value = red_image_rgb.getpixel((i[0], i[1]))
    if rgb_pixel_value == (38, 115, 0):
        verde = verde+1
    elif rgb_pixel_value == (255, 0, 0):
        vermelho = vermelho+1
    elif rgb_pixel_value == (255, 170, 0):
        amarelo = amarelo+1
    counter = 1+counter

postes_livre = amarelo+verde

#info[0,x]=coordenada, info[1,x]=cor
info = []
coords = []
for y in range(2):
    linha = []
    for x in range(postes_livre):
        linha.append(0)
    info.append(linha)
    coords.append(linha)

coords = circles[0, :]

saiu = circles

coords = coords[coords[:, 0].argsort()]

x = np.array(coords[:, 0], dtype=np.int64)
y = np.array(coords[:, 1], dtype=np.int64)

poly = lagrange(x, y)

x2 = np.linspace(0, 300)

plt.plot(x2, poly(x2), label="Polynom")


print(poly)

plt.scatter(x, y)

plt.imshow(im)
plt.show()