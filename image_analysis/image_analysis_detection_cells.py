# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:54:53 2023

@author: javie

"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog
import random
from PIL import Image
from tqdm import tqdm
import os
import cv2
import numpy as np
import math
from skimage import io




'''
This Python Script (image_analysis_detetion_cells.py) does the following:
    
    1. Asks the user for the first and last images to be analyzed.
    As the images are numerated like -######, it automatically detects all images from first to last,
    excluding numerated missing images. All images must be in the same folder.
    
    It also asks the user for the folder where the processed images will be saved.
    (it is recommended to create a separate folder)
    
    2. The sript is prepared for different types of cantilever. 
    Just enter the number corresponding to the current cantilever.
    
    3. It makes adjustments to the image:
        - Rotates the image so that the axis are completelly vertical and horizontal.
        - Detects and draw lines at the axis.
        - Draws the shape of the cantilever.
        
    4. Cell detection.
        - Please adjusts the thresholds at the function cv2.HoughCircles(), loc.475.
        - Try to adjust the thresholds such that more circles are detected. The scripts allows for user selection.
        - A window appears per image, showing all detected circles. 
        - If the circle recognizes the cell press "k". Otherwise just press any other key to pass onto next detected circle.
        - To move onto the next image press "Esc".
        
    5. The program creates and exports a GIF with the selected detections to the folder selected by the user.
'''








# SELECT THE IMAGES YOU WANT TO ANALYZE

root = Tk()
root.withdraw()

print("Please select the first image:")
first_image = filedialog.askopenfilename(filetypes=[("Image files", "*.tif")])
print("Please select the last image:")
last_image = filedialog.askopenfilename(filetypes=[("Image files", "*.tif")])    


directory_last, filename_last = os.path.split(last_image)
directory_first, filename_first = os.path.split(first_image)

image_name = str(filename_first.split("-")[0])
first_image_number = int(filename_first.split("-")[1].split(".")[0])
last_image_number = int(filename_last.split("-")[1].split(".")[0])
    
    
image_paths = []

for i in range(first_image_number, last_image_number + 1):
    image_names = f"{image_name}-{str(i).zfill(6)}.tif"  # Create image name with leading zeros
    image_path = os.path.join(directory_last, image_names)
    
    if os.path.exists(image_path):
        image_paths.append(image_path)
    
    
    
position_circles = []

GIF_images = []
    
print("")
    
def cantilever():
    print("Please Select your Cantilever.")
    print("#1. n-Ambition-5")
    print("#2. n-Ambition-10")
    
    choice = int(input("Enter your #:"))
    if choice == 1:
        len_cant = 390
    if choice == 2:
        len_cant = 590
        
    return len_cant



zero_yaxis = 15
length_cantilever = cantilever() + zero_yaxis

    





'''SAVE THE DETECTED IMAGES IN REALTIME FOLDER'''
print("Please select folder where the 'detected' images will be saved.")
folder_detected_path = filedialog.askdirectory(title="Select the folder to save the 'detected' files")

    
    
for image_path in image_paths: 
    
    if not os.path.exists(image_path):
        print(f"Image path does not exist: {image_path}")
        continue
    
    '''
    We obtain different images:
        
        IMAGE: BGR Original Image
        GRAY_IMAGE: Original Image in Grayscale
        
        COMBINED_IMAGE: BRG Image of White and Black mask of the Cantilever
        MASK_GRAY_IMAGE: GrayScale Image of the Mask
        MASK_COLOR_IMAGE: BGR Image of the Mask
        
        OUTPUT_IMAGE: The Image where we draw things in Grayscale.
    '''
    
    
    
    time = image_path[-7:-4]
           
    image = cv2.imread(image_path)
    
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
            # Apply thresholding to create a binary mask
    threshold_value =60
    _, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    # Create an all-white image and apply the binary mask
    white_regions = np.full_like(image, (255, 255, 255))
    white_regions = cv2.bitwise_and(white_regions, white_regions, mask=binary_mask)
    
    # Invert the binary mask
    inverted_mask = cv2.bitwise_not(binary_mask)
    
    # Create an all-black image and apply the inverted mask
    black_regions = np.zeros_like(image)
    black_regions = cv2.bitwise_and(black_regions, black_regions, mask=inverted_mask)
    
    # Combine the white and black regions
    combined_image = cv2.add(white_regions, black_regions)
    
    
    output_image = image
    mask_gray_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)    
    mask_color_image = cv2.cvtColor(mask_gray_image, cv2.COLOR_GRAY2BGR)
    
    
    
    
    
    
    
    
    '''
    FIND THE TOP AND BOTTOM WHITE PIXELS.
    '''
    edges = cv2.Canny(mask_gray_image, threshold1=100, threshold2=100)
    edges_try = cv2.Canny(mask_gray_image, threshold1=100, threshold2=100)
    
    
    
    
    # Define the ranges for top_rows and bottom_rows
    top_row_start = 20
    top_row_end = 100
    bottom_row_start = -100
    bottom_row_end = -20
    
    
    length_roi = 300
    
    
    top_rows = edges[top_row_start:top_row_end, :length_roi]
    bottom_rows = edges[bottom_row_start:bottom_row_end, :length_roi]   

    ''' AVERAGE X POSITION OF WHITE PIXELS IN ROI'''
    
    white_positions_top = np.argwhere(top_rows == 255)
    white_positions_bottom = np.argwhere(bottom_rows == 255)
        
        # Calculate the average x-positions for top ROI
    if white_positions_top.shape[0] > 0:
        x_positions_top = white_positions_top[:, 1]  # Extract x-positions
        avg_x_top = np.mean(x_positions_top)
    else:
        avg_x_top = None
    
    # Calculate the average x-positions for bottom ROI
    if white_positions_bottom.shape[0] > 0:
        x_positions_bottom = white_positions_bottom[:, 1]  # Extract x-positions
        avg_x_bottom = np.mean(x_positions_bottom)
    else:
        avg_x_bottom = None
        
    
    
    
    # Create the requested (x, y) positions
    y_top = (avg_x_top, 0)
    y_bot = (avg_x_bottom, edges.shape[0] - 1)
    
    
    
    
    
    
    
    
    '''CALCULATE THE ANGLE OF THE Y AXIS SO THAT THE IMAGE CAN BE ROTATED'''
    # Calculate the angle
    dy = y_bot[0] - y_top[0]
    dx = y_bot[1] - y_top[1]
    # Calculate the angle in radians
    angle_rad = np.arctan2(dy, dx)
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    rows, cols, _ = output_image.shape
    
    
    
    
    
    
    
    '''rotate the images'''
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle_deg, 1)
    # Perform the rotation
    output_image = cv2.warpAffine(output_image, rotation_matrix, (cols, rows))
    edges = cv2.warpAffine(edges, rotation_matrix, (cols, rows)) # It transforms it into grayscale
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
    
    
    
    
    ''' DRAW AVERAGE WHITE PIXEL ROI'''  
    cv2.rectangle(edges_color, (0, top_row_start), (length_roi, top_row_end), (255, 0, 0), 1)  # Green rectangle for top_rows
    cv2.rectangle(edges_color, (0, edges.shape[0] + bottom_row_start), (length_roi, edges.shape[0] + bottom_row_end), (255, 0, 0), 1)  # Red rectangle for bottom_rows

    
    
    
    
    
    
    '''DRAW THE Y-AXIS LINE'''        
    new1_top_white_pixel = np.dot(rotation_matrix, np.array([y_top[0], y_top[1], 1])).astype(int)[:2]
    new1_bottom_white_pixel = np.dot(rotation_matrix, np.array([y_bot[0], y_bot[1], 1])).astype(int)[:2]
    
    moved_top_white_pixel = (new1_top_white_pixel[0]-zero_yaxis, new1_top_white_pixel[1])
    moved_bottom_white_pixel = (new1_bottom_white_pixel[0]-zero_yaxis, new1_bottom_white_pixel[1])

    # Draw a line between the two white pixels
    line_color = (0, 255, 0)  # Green color in BGR format
    line_thickness = 1
    cv2.line(output_image,
            moved_top_white_pixel, moved_bottom_white_pixel,
            line_color, line_thickness)
    
    
    cv2.line(edges_color,
            moved_top_white_pixel, moved_bottom_white_pixel,
            (0,255,255), line_thickness)
    
    
    new_top_white_pixel = moved_top_white_pixel
    new_bottom_white_pixel = moved_bottom_white_pixel
    
    
    
    
    
    
    
    
    
    '''Look for the Cantilever edge by counting the rows with less black pixels. '''
    non_zero_pixel_counts = np.count_nonzero(edges, axis=1)

    # Find the indices of the rows with the most non-zero pixels
    significant_rows_indices = np.argsort(non_zero_pixel_counts)[::-1]
    
    # Find two indices that satisfy the minimum vertical distance constraint
    selected_indices = []
    for idx in significant_rows_indices:
        if all(abs(idx - selected_idx) >= 50 for selected_idx in selected_indices):
            selected_indices.append(idx)
        if len(selected_indices) == 2:
            break
    
    def order_integers(int_list):
        if int_list[0] > int_list[1]:
            int_list[0], int_list[1] = int_list[1], int_list[0]
        
    order_integers(selected_indices)
    # Draw horizontal lineskk on the original image for the selected rows
    for row_index in selected_indices:
        cv2.line(edges_color, (0, row_index), (image.shape[1], row_index), (200, 0, 200), 1)  # Draw an violet line
    
    
    
    '''Red line between the edges of the cantilever '''
    red_line_row = (selected_indices[0] + selected_indices[1]) // 2
    cv2.line(edges_color, (0, red_line_row), (image.shape[1], red_line_row), (0, 255, 0), 1)  # Draw a green line x-axis
    cv2.line(output_image, (0, red_line_row), (image.shape[1], red_line_row), (0, 255, 0), 1)  # Draw a green line x-axis

        
   
    
    
    
    
    
    
    
    
    '''From the rotated image, select a ROI along the Y-axis.'''
    width_ROI = 80
    
    top_left_roi = (new_top_white_pixel[0] - width_ROI//2 + zero_yaxis , width_ROI)
    top_right_roi = (new_top_white_pixel[0] + width_ROI//2 + zero_yaxis, width_ROI)
    bottom_left_roi = (new_bottom_white_pixel[0] - width_ROI // 2 + zero_yaxis, output_image.shape[0] - width_ROI)
    bottom_right_roi = (new_bottom_white_pixel[0] + width_ROI // 2 + zero_yaxis, output_image.shape[0] - width_ROI)
    
    
    
    roi = edges[top_left_roi[1]:bottom_left_roi[1] + 1, top_left_roi[0]:top_right_roi[0] + 1]
    roi_color = (0, 0, 255)  # Red color in BGR format
    roi_thickness = 2
    
    
    '''Compute the middle index row of the cantilever'''
    # Find rows where all elements are zeros
    zero_rows_indices = np.all(roi == 0, axis=1)
    # Extract the row numbers of rows with no gray pixels
    rows_with_no_gray_pixels_indices = np.where(zero_rows_indices)[0] + width_ROI
    x_axis_mean = int(np.mean(rows_with_no_gray_pixels_indices))
    x_axis_median = int(np.median(rows_with_no_gray_pixels_indices))
    
    # IT DOES NOTHING. JUST SAVES THE X-AXIS MEAN AND MEDIAN FROM THE BLACK ROWS OF THE RED ROI.
    
    
    
    
    
    
    

    '''Draw the red ROI'''
    cv2.rectangle(edges_color, top_left_roi, bottom_right_roi, roi_color, 1)
            
    #Y-axis on edges_color
    cv2.line(edges_color, 
            (new_top_white_pixel[0] + zero_yaxis, new_top_white_pixel[1]),
            (new_bottom_white_pixel[0] + zero_yaxis, new_bottom_white_pixel[1]),
            line_color, line_thickness)
    
    
    
    
    
    
        # Calculate the intersection point of the two lines
    origin_coordinates = (new_top_white_pixel[0], red_line_row)
    text = "(0,0)"
    
    # Define font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White color
    font_thickness = 1
    
    # Get the size of the text
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size
    
    # Calculate the position to place the text
    text_position = (origin_coordinates[0] - text_width-10, origin_coordinates[1] - text_height-5)
    
    # Draw the text on the image
    cv2.putText(output_image, text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(edges_color, text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    
    
    # Draw the origin marker (circle or dot)
    marker_radius = 4
    marker_color = (218, 112, 214)  # Green color for the marker
    cv2.circle(edges_color, origin_coordinates, marker_radius, marker_color, -1)  # -1 to fill the circle
    cv2.circle(output_image, origin_coordinates, marker_radius, marker_color, -1)
       





    
    
    '''COMPUTE THE EDGE OF THE CANTILEVER TO NORMALIZE LENGTHS.
        n-ambition-5: 390
        n-Ambition-10. 590
    '''
    
    
    
    edge_cantilever = (origin_coordinates[0] + length_cantilever, origin_coordinates[1])
    
    cv2.circle(output_image, edge_cantilever, marker_radius, marker_color, -1)
    cv2.circle(edges_color, edge_cantilever, marker_radius, marker_color, -1)
    
    text_position_edge = (edge_cantilever[0] - 15, edge_cantilever[1] + text_height + 10)
    cv2.putText(output_image, "(1,0)", text_position_edge, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(edges_color, "(1,0)", text_position_edge, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    
    
    
    
    cv2.line(edges_color, (edge_cantilever[0], 0), (edge_cantilever[0], output_image.shape[0]), line_color, line_thickness)
    
    
    
    
    
    
    # CANTILEVER SHAPE
    L1 = 340 + zero_yaxis
    b2 = 50

    #DRAW POINTS OF THE CANTILEVER
    c1 = (origin_coordinates[0],selected_indices[0])
    c2 = (origin_coordinates[0] + L1, selected_indices[0])
    c3 = (edge_cantilever[0], edge_cantilever[1]- int(b2/2))
    c4 = (edge_cantilever[0], edge_cantilever[1]+ int(b2/2))
    c5 = (origin_coordinates[0] + L1, selected_indices[1])
    c6 = (origin_coordinates[0],selected_indices[1])
    
    polygon_vertices = [c1, c2, c3, c4, c5, c6]
    
    
    
     
    
    '''
    DRAW THE CIRCLES FROM THE ORIGINAL (GRAYSCALE) IMAGE INTO THE OUTPUT_IMAGE
    '''
    output_image_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)  
    
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    
    
    circles = cv2.HoughCircles(
        output_image_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=110,      # CHANGE THIS PARAMETER, EDGE DETECTION
        param2=18,      # THRESHOLD FOR DETECTION, LESS DETECTIONS = HIGHER NUMBER
        minRadius=50,
        maxRadius=120
    )
    
    
    # Convert the (x, y) coordinates and radius of the circles to integers
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        display_image = output_image.copy()
        # Draw detected circles on the original image
        for (x, y, r) in circles:
            
            cv2.circle(display_image, (x, y), r, (0, 255, 255), 1)  # Yellow circle outline
            cv2.circle(display_image, (x, y), 2, (0, 0, 255), 2)    # Draw center point
            
            
        for (x,y,r) in circles:
            display_image2 = display_image.copy()
            cv2.circle(display_image2, (x, y), r, (0, 0, 255), 2)  # GREEN circle outline
            cv2.circle(display_image2, (x, y), 2, (0, 255, 0), 2)    # Draw center point
            cv2.imshow("Detected Circle. Press K to save the detected circle.", display_image2)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord("k"):
                # Keep the detected circle
                cv2.circle(output_image, (x, y), r, (0, 255, 255), 1)  # Yellow circle outline
                cv2.circle(output_image, (x, y), 2, (0, 0, 255), 2)    # Draw center point
                cv2.circle(edges_color, (x, y), r, (0, 255, 255), 2)  # Yellow circle outline
                cv2.circle(edges_color, (x, y), 2, (0, 0, 255), 2)    # Draw center point
                
                
                
                mask_circle = np.zeros((768, 1024), dtype=np.uint8)
                cv2.circle(mask_circle, (x,y), r, 255, thickness=cv2.FILLED)
                mask_polygon = np.zeros((768, 1024), dtype=np.uint8)
                cv2.fillPoly(mask_polygon, [np.array(polygon_vertices)], 255)
            
                intersection = cv2.bitwise_and(mask_circle, mask_polygon)

                # Count the non-zero pixels in the intersection mask
                intersection_area = np.count_nonzero(intersection)
                
                
                # Create a yellow overlay with transparency
                overlay = edges_color.copy()
                overlay[intersection != 0] = (0, 255, 255)  # Yellow color
            
                # Apply the overlay with some transparency (adjust the alpha value)
                alpha = 0.1
                cv2.addWeighted(overlay, alpha, edges_color, 1 - alpha, 0, edges_color)
                            

                
                coordinates_circles = (time, 
                                   (x-origin_coordinates[0])/length_cantilever, 
                                   (origin_coordinates[1]-y)/length_cantilever, 
                                   r/length_cantilever,x,y,r, intersection_area)
            
                text_position_cell = f"({coordinates_circles[1]:.2f}, {coordinates_circles[2]:.2f})"
            
                position_circles.append(coordinates_circles)
                
                cv2.putText(output_image, text_position_cell, (x-100,y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.putText(output_image, text_position_cell, (x-100,y), font, font_scale, font_color, font_thickness, cv2.LINE_AA) 
                
            elif key == 27:  #Press ESC to finish selecting Circles
                break
            
            
        cv2.destroyAllWindows()
            
            
            

    '''
    SCALE OF THE IMAGE
    '''
    
    scale_length_pixels = 100  # Length of the scale in pixels
    scale_position = (image.shape[1] - scale_length_pixels - 20, image.shape[0] - 20)  # Bottom right corner
    scale_end = (scale_position[0] + scale_length_pixels, scale_position[1])
    
    cv2.line(output_image, scale_position, scale_end, (0, 255, 255), 2)  # Draw a blue line for the scale
    cv2.line(edges_color, scale_position, scale_end, (0, 255, 255), 2)  # Draw a blue line for the scale
    
    # Add text indicating the scale length
    scale_text = f'{scale_length_pixels} pixels'
    cv2.putText(output_image, scale_text, (scale_position[0], scale_position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(edges_color, scale_text, (scale_position[0], scale_position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    
    ''' ADD TIME STAMP'''
    
    cv2.putText(output_image, f"time={time} min",  
                (scale_position[0], scale_position[1] - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(edges_color, f"time={time} min",(scale_position[0], scale_position[1] - 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)






    ''' DRAW EVERYTHING IN OUTPUT_IMAGE (GRAY)'''
    
    # MIDDLE X AXIS
    cv2.line(output_image, (0, red_line_row), (image.shape[1], red_line_row), (0, 255, 0), 1)  # Draw a green line x-axis

    # Y_AXIS
    cv2.line(output_image, new_top_white_pixel, new_bottom_white_pixel, line_color, line_thickness)
    
    # TIME STAMP
    cv2.putText(output_image, f"time={time} s", (scale_position[0], scale_position[1] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # SCALE
    cv2.putText(output_image, scale_text, (scale_position[0], scale_position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(edges_color, scale_text, (scale_position[0], scale_position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.line(output_image, scale_position, scale_end, (0, 255, 255), 2)  # Draw a blue line for the scale
    cv2.line(edges_color, scale_position, scale_end, (0, 255, 255), 2)  # Draw a blue line for the scale
    
    
    # LENGTH CANTILEVER
    cv2.putText(output_image, "(1,0)", text_position_edge, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.circle(output_image, edge_cantilever, marker_radius, marker_color, -1)


    # ORIGIN COORDINATES
    cv2.circle(output_image, origin_coordinates, marker_radius, marker_color, -1)
    cv2.putText(output_image, text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    

    
    
    ''' DRAW THE SHAPE OF THE CANTILEVER '''
    cv2.circle(edges_color, (origin_coordinates[0],selected_indices[0]), marker_radius, marker_color, -1)
    cv2.circle(edges_color, (origin_coordinates[0] + L1, selected_indices[0]), marker_radius, marker_color, -1)
    
    cv2.line(edges_color, c1, c2, (0,140,255), 1) 
    cv2.line(output_image, c1, c2, (0,140,255), 1) 
    
    
    
    cv2.circle(edges_color, (origin_coordinates[0],selected_indices[1]), marker_radius, marker_color, -1)
    cv2.circle(edges_color, (origin_coordinates[0] + L1, selected_indices[1]), marker_radius, marker_color, -1)
    
    cv2.line(edges_color, c6, c5, (0,140,255), 1) 
    cv2.line(output_image, c6, c5, (0,140,255), 1) 


    cv2.circle(edges_color, (edge_cantilever[0], edge_cantilever[1]- int(b2/2)), marker_radius, marker_color, -1)
    cv2.circle(edges_color, (edge_cantilever[0], edge_cantilever[1]+ int(b2/2)), marker_radius, marker_color, -1)
    
      
    cv2.line(edges_color,c3,c4,(0,140,255), 1)  
    cv2.line(edges_color,c3,c2,(0,140,255), 1)  #    
    cv2.line(edges_color, c4,c5,(0,140,255), 1)      
    cv2.line(edges_color, c1,c6, (0,140,255), 1)  
    
    
    cv2.line(output_image, c3,c4, (0,140,255), 1)  
    cv2.line(output_image, c3,c2, (0,140,255), 1)  #    
    cv2.line(output_image,  c4,c5,(0,140,255), 1)     
    cv2.line(output_image, c1,c6,(0,140,255), 1)  








    csv_file_path = folder_detected_path + '/position_circles.csv'
    field_names = ['time', 'x_norm', 'y_norm', 'r_norm', 'x', 'y', 'r', 'area']

    df_circles = pd.DataFrame(position_circles, columns = field_names)
    
    
    





    
    
    
    '''
    PRINT THE OUTPUT_IMAGE
    '''
    
    combined_image = cv2.hconcat((output_image, edges_color))
    resized_image = combined_image
    
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    GIF_images.append(Image.fromarray(resized_image_rgb))
    
    # Display the resized_image (for example, using cv2.imshow())
    cv2.imshow('Resized Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    '''Save the processed image for further analysis'''
    image_filename = os.path.basename(image_path)
    detected_filename = f'{os.path.splitext(image_filename)[0]}.jpg'
    pil_image = Image.fromarray(resized_image_rgb)
    pil_image.save(os.path.join(folder_detected_path, detected_filename))
    
    
    
# Create a GIF from the list of resized images
print("Select the folder to save the GIF file.")
output_gif_folder = folder_detected_path
output_gif_filename = f'{image_name}_detected.gif'
output_gif_path = os.path.join(output_gif_folder, output_gif_filename)
GIF_images[0].save(
    output_gif_path,
    save_all=True,
    append_images=GIF_images[1:],
    loop=0,  # Loop indefinitely
    duration=300  # Duration between frames in milliseconds
)



df_circles.to_csv(csv_file_path, index=False)


print("Images Processed Successfully!!")
