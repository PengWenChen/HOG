# from matplotlib import pyplot as plt
import math
import cv2
import numpy as np
import image_preprocessing

class hog_descriptor:
    """class of calculating hog and drawing gradient functions"""
    def __init__(self, img, cell_size, block_size, overlap_num, bin_num):
        self.img = img
        self.cell_size = cell_size
        self.block_size = block_size
        self.overlap_num = overlap_num # = block stride
        self.bin_num = bin_num
        # If overlap==0, then block_size would equal to cell_size.
    def hog(self):
        """
        Calulate gradients and magnitude of each pixels. Classify them by 8*8 cell.
        Parameters:
            self
        Returns:
            histogram with 9 bins of each cells.
        Raises:
        """
        img = self.img
        cell_size = self.cell_size
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        ang = ang % 180 # the range of angle = (0,179)
        histogram = []
        for i in range(0, int(img.shape[0]/cell_size)): #0~15
            for j in range(0, int(img.shape[1]/cell_size)): #0~8
                cell_angle = []
                cell_magnitude = []
                cell_angle.append(ang[i*cell_size : i*cell_size+cell_size, j*cell_size : j*cell_size+cell_size])
                cell_magnitude.append(mag[i*cell_size : i*cell_size+cell_size, j*cell_size : j*cell_size+cell_size])
                hist = self.cal_histogram(cell_angle[0], cell_magnitude[0])
                histogram.append(hist)
        histogram = np.array(histogram)
        return histogram

    def cal_histogram(self, cell_angle, cell_magnitude):
        """
        Calculate histogram.
        Parameters:
            param1 - cell_angle vector of the cell from hog function
            param2 - cell_magnitude vector of the cell from hog function
        Returns:
            return histogram of that cell
        Raises:
        """
        bin_num = self.bin_num
        orientation_centers = [0] * bin_num 
        bin_size = int(180/bin_num)
        for k in range(cell_magnitude.shape[0]):
            for l in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[k][l]
                gradient_angle = cell_angle[k][l]
                min_angle = int(gradient_angle / bin_size) # find the bin to add into
                max_angle = (int(gradient_angle / bin_size)+1) % bin_num # find the bin to add into
                mod = (gradient_angle-min_angle*bin_size)/bin_size # calculate the ratio of gradient_strength(=magnitude)
                orientation_centers[min_angle] += (gradient_strength * (1 - mod)) # ex: 25 degree will have 75% put into bin1
                orientation_centers[max_angle] += (gradient_strength * mod)       # and 25% put into bin2
        return orientation_centers


    def draw_gradient(self, histogram, hog_img):
        """
        Draw gradient on the original img.
        Parameters:
            param1 - All cells' histogram vector.
            param2 - original img
        Returns:
            img with graient lines
        Raises:
        """
        block_size = self.block_size
        hog_img = self.img
        overlap = self.overlap_num
        bin_num = self.bin_num
        cell_x_num = int((hog_img.shape[1]-overlap) / (block_size-overlap))
        cell_y_num = int((hog_img.shape[0]-overlap) / (block_size-overlap))
        x_center = 0
        y_center = 0
        for j in range(0, cell_y_num): #0~15 cell_y_num 0~14
            for i in range(0, cell_x_num): #0~7 cell_x_num 0~6
                x_center = int(block_size/2+i*(block_size-overlap))
                y_center = int(block_size/2+j*(block_size-overlap))
                for line in range(0, bin_num):
                    line_length = histogram[i+j*cell_x_num][line]
                    # if line_length<0.8: # Use it when calculating histograms by cv2
                    #     line_length = 0
                    # else:line_length = line_length*2
                    line_length = line_length*0.3 # Use it when calculating histograms by hog_descriptor.hog()
                                                       # bolt:*0.3, cat1:*0.0015
                    angle_rad = (line*20*np.pi) / 180
                    x_end = int(x_center - (line_length)*math.sin(angle_rad)) # Lines at left hand side
                    y_end = int(y_center - (line_length)*math.cos(angle_rad)) # Lines at left hand side
                    x_end_opposite = int(x_center + (line_length)*math.sin(angle_rad)) # Lines at right hand side
                    y_end_opposite = int(y_center + (line_length)*math.cos(angle_rad)) # Lines at right hand side
                    cv2.line(hog_img, (x_end, y_end), (x_end_opposite, y_end_opposite), (0, 0, 255), 1)
                    # 0 degree points to the top, and add degrees counterclockwise.
        cv2.imshow("hog_img", hog_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return hog_img

if __name__ == '__main__':
    img_bolt = cv2.imread('./img/bolt2.png', 0)
    img_bolt = np.float32(img_bolt) / 255.0 # a way to normalize to 0~1
    img_bolt = image_preprocessing.gamma_transform(0.5, img_bolt)
    
    img_bolt = img_bolt[100:300, 200:300]
    img_bolt = cv2.resize(img_bolt, (64, 128))
    cv2.imshow("img_bolt", img_bolt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    bolt_hog = hog_descriptor(img_bolt, 8, 8, 0, 9)
    bolt_hist = bolt_hog.hog()
    new_img = bolt_hog.draw_gradient(bolt_hist, img_bolt)

    new_img = cv2.normalize(new_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Since imwrite will output all black img with pixels in range 0~1, 
    # we need to normalize it back to range 0~255
    cv2.imwrite('./img/gradient_output/bolt_calculate_myself.jpg', new_img)
