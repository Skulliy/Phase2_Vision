import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc
import glob2
import imageio

path = '../test_dataset/IMG/*'
img_list = glob2.glob(path)
index = np.random.randint(0,len(img_list)-1)
image = mpimg.imread(img_list[index])

example_grid = '../calibration_images/example_grid1.jpg'
example_rock = "../calibration_images/example_rock1.jpg"
grid_img = mpimg.imread(example_grid)
rock_img = mpimg.imread(example_rock)

fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(grid_img)
plt.subplot(122)
plt.imshow(rock_img)
plt.show()
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return warped

dst_size = 3
bottom_offset = 5
source = np.float32([[14,140],
                    [300,140],
                    [200,95],
                    [120,95]])

destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                         [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                         [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                         [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset]])

warped = perspect_transform(rock_img, source, destination)
plt.imshow(warped)
def color_thresh(img, rgb_thresh=(160,160,160)):
    color_select = np.zeros_like(img[:,:,1])
    above_thresh = (img[:,:,0] > rgb_thresh[0]) &(img[:,:,1] > rgb_thresh[1]) &(img[:,:,2] > rgb_thresh[2])
    color_select[above_thresh] = 1
    return color_select

def color_threshob(img, rgb_thresh=(120,120,120)):
    color_select = np.zeros_like(img[:,:,1])
    above_thresh = (img[:,:,0] < rgb_thresh[0]) &(img[:,:,1] < rgb_thresh[1]) &(img[:,:,2] < rgb_thresh[2])
    color_select[above_thresh] = 1
    return color_select

def color_threshr(img, rgb_thresh=(100,100,20)):
    color_select = np.zeros_like(img[:,:,1])
    above_thresh = (img[:,:,0] > rgb_thresh[0]) &(img[:,:,1] > rgb_thresh[1]) &(img[:,:,2] < rgb_thresh[2])
    color_select[above_thresh] = 1
    return color_select


threshed = color_threshob(warped)
plt.imshow(threshed, cmap='gray')
plt.show()
