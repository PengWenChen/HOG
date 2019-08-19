import numpy as np
import cv2
# ToDo(PengWen): hog_calculation & image_preprocessing are loop-calling each other, please resolve the dependency
from hog_calculation import hog_descriptor
import bolt_hog

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
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = "./img/cat1.jpg"
    img_cat_1 = cv2.imread(path)# original image size:(y:1427, x:1920, 3)
                                # You can use both grayscale and RGB when calculating by cv2.
                                # If you want to use hog_descriptor.hog(),
                                # please use with grayscale. (path,0)

    img_cat_1 = cv2.resize(img_cat_1, (640, 384)) # (parameter1:x parameter2:y),
                                                  # resize it because the img is too big
    img_cat_1 = img_cat_1[0:320, 150:470] # cut to the cat head with width:320px,height:320px
    # show_img('cat',img_cat_1)

    norm_cat_1 = cv2.normalize(img_cat_1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # show_img('norm_img',norm_cat_1)

    imgGamma_0p5 = gamma_transform(1/2, norm_cat_1)
    # show_img('imgGamma_0p5',imgGamma_0p5)
    normback_cat_1 = cv2.normalize(imgGamma_0p5, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    normback_cat_1 = np.uint8(normback_cat_1) # cv2.HOGDescriptor only read uint8 img
    # show_img('normback_cat_1', normback_cat_1)

    # Lab_cat_1 = cv2.cvtColor(normback_cat_1, cv2.COLOR_BGR2Lab) # I didn't use Lab domain.
    # show_img('Lab_cat_1', Lab_cat_1)

    #### CALCULATE BY cv2.HOGDescriptor.compute() ######
    winSize = (320, 320)  #p1:x, p2:y
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog2 = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hist_cat = hog2.compute(normback_cat_1)

    hist_cat = np.reshape(hist_cat, (int(hist_cat.shape[0]/36), 36)) # reshape hist_cat with four histograms(=block size)
    new_hist = bolt_hog.cal_histogram_2(hist_cat, nbins)  # reorganize the final histograms (4 cells per new histogram)
                                                         # using cal_histogram_2 func from bolt_hog.py
    h_obj = hog_descriptor(normback_cat_1, 8, 16, 8, 9)  # create a hog_descriptor obj
    cat_1 = hog_descriptor.draw_gradient(h_obj, new_hist, normback_cat_1) # use the draw_gradient method of the obj
    cv2.imwrite('./img/gradient_output/cat1_cv2.jpg', cat_1) # save the output
    # cat_1 = cv2.resize(cat_1, (int(cat_1.shape[1]*3), int(cat_1.shape[0]*3))) #zoom in


    #### CALCULATE BY hog_descriptor.hog() ######
    # h_obj = hog_descriptor(normback_cat_1, 8, 8, 0, 9)
    # new_hist = h_obj.hog()
    # cat_1 = h_obj.draw_gradient(new_hist, img_cat_1)
    # cv2.imwrite('./img/gradient_output/cat1_calculate_myself.jpg', cat_1)
