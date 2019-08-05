import numpy as np
from matplotlib import pyplot as plt
import cv2

def gamma_transform(gamma, img):
    """
    This is a gamma correction function.
    Parameters:
        param1 - the power number
        param2 - img that needs to do gamma correction
    Returns:
        img
    Raises:
    """
    img = np.power(img, gamma)
    return img

def show_img(name, img):
    """
    This is a showing img function for the convience of saving time typing imshow, waitKey...etc.
    Parameters:
        param1 - the window name
        param2 - the img that you want to show
    Returns:
    Raises:
    """
    if img.shape[0] > 1300:
        re_y = int(img.shape[0]/3)
        re_x = int(img.shape[1]/3)
        img = cv2.resize(img, (re_x, re_y))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_hist(hist):
    """
    This is a drawing-histogram function.
    Parameters:
        param1 - the vector that hog computes out
    Returns:
    Raises:
    """
    y_value = np.zeros(9)
    for j in range(0, hist.shape[0], 9):
        hist_out = np.array(hist[j:j+9]).tolist()
        y_temp = []
        for i in hist_out:
            y_temp.append(i[0])
        y_value += np.array(y_temp)
        # print(y_value)
    plt.bar([0, 20, 40, 60, 80, 100, 120, 140, 160], y_value, width=10)
    plt.savefig("GradientHistogram.png")
    plt.show()


path = "./cat1.jpg"
img_1 = cv2.imread(path)
dst = np.zeros(img_1.shape)
print(img_1.shape)
imgNorm_1 = cv2.normalize(img_1, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

imgGamma_5 = gamma_transform(0.5, imgNorm_1)
imgGamma_1 = gamma_transform(1, imgNorm_1)
imgGamma_2 = gamma_transform(2, imgNorm_1)

imgLab = cv2.cvtColor(imgGamma_1, cv2.COLOR_BGR2LAB)
show_img('imgLab', imgLab)


img_bolt = cv2.imread('./bolt2.png', 0)
x = 200 # 裁切區域的 x 與 y 座標（左上角）
y = 100
w = 100 # 裁切區域的長度與寬度
h = 200 # 裁切圖片
img_bolt = img_bolt[y:y+h, x:x+w]
img_bolt = cv2.resize(img_bolt, (64, 128))
cv2.imshow('bolt_resized', img_bolt)
cv2.waitKey(0)
cv2.destroyAllWindows()

winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
hist_bolt = hog.compute(img_bolt)
hist_cat1 = hog.compute(imgLab)
draw_hist(hist_bolt)
