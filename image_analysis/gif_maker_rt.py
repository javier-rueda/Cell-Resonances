


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog
import random
from PIL import Image
from tqdm import tqdm
import os

try:
    root = Tk()
    root.withdraw()
    
    print("Please select the first image:")
    first_image = filedialog.askopenfilename(filetypes=[("Image files", "*.tif")])
    print("Please select the last image:")
    last_image = filedialog.askopenfilename(filetypes=[("Image files", "*.tif")])
    
    directory_last, filename_last = os.path.split(last_image)
    directory_first, filename_first = os.path.split(first_image)
    save_gif_directory = r'C:\Users\javie\OneDrive\Escritorio\Experiment Data AFM\Images\GIFS'
    
    
    image_name = str(filename_first.split("-")[0])
    first_image_number = int(filename_first.split("-")[1].split(".")[0])
    last_image_number = int(filename_last.split("-")[1].split(".")[0])
    
    
    image_paths = []
    
    for i in range(first_image_number, last_image_number + 1):
        image_names = f"{image_name}-{str(i).zfill(6)}.tif"  # Create image name with leading zeros
        image_path = os.path.join(directory_last, image_names)
        image_paths.append(image_path)
            
            
    print("Creating RGB - GIF file, please wait...")
    
    
    frames = []
    for i, image_path in tqdm(enumerate(image_paths), desc="Processing images", unit="image"):
        try:
            image = Image.open(image_path)
            frames.append(image)
            
            # Check if there is a next image
            if i + 1 < len(image_paths):
                next_image = Image.open(image_paths[i + 1])
                frames.append(next_image)
    
        except Exception as e:
            print(f"Error processing image: {image_path}. Skipping to the next image.")
            continue
    
    # Save the frames as a GIF file
    if len(frames) > 0:
        gif_path = os.path.join(save_gif_directory, f"{image_name}_RGB.gif")
        frames[0].save(
            gif_path,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=100,  # Set the duration (in milliseconds) between frames
            loop=0,       # Set loop to 0 for an infinite loop, or any other number for a limited loop
        )
        
    print("RGB GIF file created successfully!")
    
    print("")
    print("Creating Grayscale - GIF file, please wait...")

    gif_image = Image.open(gif_path)
    gif_path_gray = os.path.join(save_gif_directory, f"{image_name}_grayscale.gif")
    num_frames = gif_image.n_frames
    gray_frames = []
    
    for frame_num in tqdm(range(num_frames), desc="Converting to grayscale", unit="frame"):
        gif_image.seek(frame_num)
        gray_frame = gif_image.convert("L")
        gray_frames.append(gray_frame)
        
        gray_frames[0].save(gif_path_gray, save_all=True, append_images=gray_frames[1:], loop=0, duration=gif_image.info['duration'])
        
    print("Grayscale GIF created successfully!")
    

except Exception as e:
    print(f"An error occurred: {e}")
