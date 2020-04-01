"""
Created on Wed Jun  6 23:08:15 2018

@author: bakhtiar
"""
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

import os

def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

def canny(img,low_threshold,high_threshold):
    return cv2.Canny(img,low_threshold,high_threshold)

def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def region_of_interest(img,vertices):
    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def draw_lines(img,lines,color=[255,0,0],thickness=5):
    
    negative_slopes = []
    negative_intercepts = []
    positive_slopes = []
    positive_intercepts = []
    left_points_x = []
    left_points_y = []
    right_points_x = []
    right_points_y = []
    
    y_max = img.shape[0]
    y_min = img.shape[0]
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            m = (y2-y1)/(x2-x1)
            
            if m < 0.0 and m > -math.inf:
                negative_slopes.append(m)
                left_points_x.append(x1)
                left_points_x.append(x2)
                left_points_y.append(y1)
                left_points_y.append(y2)
                negative_intercepts.append(y1-m*x1)
                
            if m > 0.0 and m < math.inf:
                positive_slopes.append(m)
                right_points_x.append(x1)
                right_points_x.append(x2)
                right_points_y.append(y1)
                right_points_y.append(y2)
                positive_intercepts.append(y1-m*x1)
                
            y_min = min(y1,y2,y_min)
            
            
    if len(negative_slopes) > 0:
        left_slope = np.mean(negative_slopes)
        left_intercept = np.mean(negative_intercepts)
        x_min_left = int((y_min - left_intercept)/left_slope)     #x = (y - b)/m
        x_max_left = int((y_max - left_intercept)/left_slope)        
        cv2.line(img,(x_min_left,y_min),(x_max_left,y_max),(255,0,0),8)
    
    if len(positive_slopes) > 0:
        right_slope = np.mean(positive_slopes)
        right_intercept = np.mean(positive_intercepts)
        x_min_right = int((y_min - right_intercept)/right_slope)
        x_max_right = int((y_max - right_intercept)/right_slope)        
        cv2.line(img,(x_min_right,y_min),(x_max_right,y_max),(255,0,0),8)                        


def hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):
    
    lines = cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),minLineLength=min_line_len,maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    draw_lines(line_img,lines)
    return line_img


for image_name in os.listdir("test_images/"):
 #   if idx==0:
 #       continue
    
    print(image_name)
    image = mpimg.imread('test_images/' + image_name)
    gray = grayscale(image)
    
    kernel_size = 5
    blur_gray = gaussian_blur(gray,kernel_size)
    
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray,low_threshold,high_threshold)   
    
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450,325),(500,325),(imshape[1],imshape[0])]],dtype=np.int32)
    masked_edges = region_of_interest(edges,vertices)
    
    rho = 1
    theta = np.pi/180
    threshold_check = 15
    min_line_length = 30
    max_line_gap = 10
    
    line_image = hough_lines(masked_edges,rho,theta,threshold_check,min_line_length,max_line_gap)
    color_edges = np.dstack((edges,edges,edges))

#    lines_edges = cv2.addWeighted(color_edges,0.8,line_image,1,0)    
    lines_edges = cv2.addWeighted(image,0.8,line_image,1,0)
    plt.imshow(lines_edges)
    plt.show()
    
    #RGB_final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    
#    mpimg.imsave(image_name+"_line_edge.jpg",lines_edges)