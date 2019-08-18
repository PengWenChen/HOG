import numpy as np
import cv2
from hog_calculation import hog_descriptor

def cal_histogram_2(histogram, bin_num):
    """
    Reorganize histograms by bin_num from the 2-D array(105,36) calculated by cv2.HOGDescriptor
    Parameters:
        param1 - the 2-D array
        param2 - number of bins
    Returns:
        2-D array(105,9) of histograms with bin_num
    Raises:
    """
    new_histogram = []
    for i in range(0, histogram.shape[0]):
        tmp = [0] * bin_num
        for j in range(0, histogram.shape[1]):
            tmp[j % bin_num] = tmp[j % bin_num] + histogram[i][j]
        new_histogram.append(tmp)
    return new_histogram

if __name__ == '__main__':
    #calculate by cv2
    img = cv2.imread('./img/bolt2.png')

    img_bolt2 = img[100:300, 200:300]
    img_bolt2 = cv2.resize(img_bolt2, (64, 128))

    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog2 = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hist_bolt = hog2.compute(img_bolt2)
    hist_bolt = np.reshape(hist_bolt, (105, 36)) # reshape hist_bolt with four histograms(=block size)
    new_hist = cal_histogram_2(hist_bolt, nbins) # reorganize the final histograms (4 cells per new histogram)
                                                 # using cal_histogram_2 func from bolt_hog.py
    h_obj = hog_descriptor(img_bolt2, 8, 16, 8, 9) # create a hog_descriptor obj
    img_output = hog_descriptor.draw_gradient(h_obj, new_hist, img_bolt2) # use the draw_gradient method of the obj
    cv2.imwrite('./img/gradient_output/bolt_cv2.jpg', img_output) # save the output
    img = cv2.resize(img_output, (256, 384)) #zoom in
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
