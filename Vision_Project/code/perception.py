import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc
import glob2
import imageio
import os
# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 200
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst,):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def color_threshrock(img, rgb_thresh=(100,100,20)):
    color_select = np.zeros_like(img[:,:,1])
    above_thresh = (img[:,:,0] > rgb_thresh[0]) &(img[:,:,1] > rgb_thresh[1]) &(img[:,:,2] < rgb_thresh[2])
    color_select[above_thresh] = 200
    return color_select

def color_threshob(img, rgb_thresh=(105,105,105)):
    color_select = np.zeros_like(img[:,:,1])
    above_thresh = (img[:,:,0] < rgb_thresh[0]) &(img[:,:,1] < rgb_thresh[1]) &(img[:,:,2] < rgb_thresh[2])
    color_select[above_thresh] = 200
    return color_select
    
def limit_range(x,y,range = 80):
    dist= np.sqrt(x**2 + y**2)
    return x[dist < range], y[dist < range]

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover,debug = False):
    dst_size = 3
    bottom_offset = 5
    source = np.float32([[14,140],
                    [300,140],
                    [200,95],
                    [120,95]])

    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                         [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                         [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                         [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset]])
    if(debug):
        original = Rover.img
        cv2.imwrite("debug/" + str(Rover.total_time) + "Original.jpg",original)
        warped = perspect_transform(original, source, destination)
        cv2.imwrite("debug/" + str(Rover.total_time) + "warped.jpg",warped)
        navigable = color_thresh(warped)
        cv2.imwrite("debug/" + str(Rover.total_time) + "nav.jpg",navigable)
        rock_samples = color_threshrock(warped)
        cv2.imwrite("debug/" + str(Rover.total_time) + "rock.jpg",rock_samples)
        obstacles = color_threshob(warped)
        cv2.imwrite("debug/" + str(Rover.total_time) + "obstacles.jpg",obstacles)
    else:
        original = Rover.img
        warped = perspect_transform(original, source, destination)
        navigable = color_thresh(warped)
        rock_samples = color_threshrock(warped)
        obstacles = color_threshob(warped)

    Rover.vision_image[:,:,0] = obstacles*255 
    Rover.vision_image[:,:,1] = rock_samples*255
    Rover.vision_image[:,:,2] = navigable*255

    x, y = rover_coords(navigable)
    x, y = limit_range(x,y)
    dst, angles = to_polar_coords(x, y)
    Rover.nav_dists = dst
    Rover.nav_angles = angles
    
    x1,y1 = rover_coords(rock_samples)
    dst, angles = to_polar_coords(x1, y1)
    
    #getting sample angles and distance
    Rover.samples_dists = dst
    Rover.samples_angles = angles
    Rover.rock_angle = np.mean(Rover.samples_angles * 180 / np.pi)
    navigable_x, navigable_y = pix_to_world(x,y,Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0],6)
    rock_x,rock_y = pix_to_world(x1,y1,Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0],6)
    Rover.rock_dist = np.mean(Rover.samples_dists)

    x2,y2 = rover_coords(obstacles)
    x2,y2 = limit_range(x2,y2)
    o_x,o_y = pix_to_world(x2,y2,Rover.pos[0],Rover.pos[1],Rover.yaw,Rover.worldmap.shape[0],6)
    
    if (Rover.pitch < 1 or Rover.pitch > 359) and (Rover.roll < 1 or Rover.roll > 359):
        Rover.worldmap[o_y, o_x, 0] = 255
        Rover.worldmap[rock_y, rock_x,1] = 255
        Rover.worldmap[navigable_y, navigable_x, 2] = 255
        # remove overlap mesurements
        nav_pix = Rover.worldmap[:, :, 2] > 0
        Rover.worldmap[nav_pix, 0] = 0
        # clip to avoid overflow
        Rover.worldmap = np.clip(Rover.worldmap, 0, 255)
        Rover.worldmap[o_y, o_x, 0] += 1
        Rover.worldmap[rock_y, rock_x, 1] += 1
        Rover.worldmap[navigable_y, navigable_x, 2] += 1
    return Rover
