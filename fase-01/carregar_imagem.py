import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread("fase-01/images/face.jpg")

inicio = (50, 5)
fim = (220, 160)
cor = (255, 0, 0)
espessura = 2

imagem_com_retangulo = cv2.rectangle(imagem.copy(), inicio, fim, cor, espessura )
imagem_com_retangulo_rgb = cv2.cvtColor(imagem_com_retangulo, cv2.COLOR_BGR2RGB)
plt.imshow(imagem_com_retangulo_rgb)
plt.axis("off")
plt.show()