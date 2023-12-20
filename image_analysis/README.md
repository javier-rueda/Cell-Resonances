# Image Analysis for Cell Detection

## Overview

The "image_analysis" directory hosts a Python script aimed at automating the detection of cells in Atomic Force Microscopy (AFM) images. The script encompasses several key functionalities to enhance the accuracy and efficiency of cell identification:

1. **Orientation Detection:**
   - The script begins by automatically identifying the orientation of the cantilever in the AFM image.
   - Subsequently, it rotates the image to align the cantilever perpendicular to a standardized reference direction.

2. **Cell Detection:**
   - The core functionality involves the use of circle detection algorithms to pinpoint potential cell locations within the image.

3. **User Selection:**
   - To ensure precision, the script engages the user in the identification process.
   - The user is prompted to manually select circles that are most fitting for enclosing cells.

4. **Output:**
   - Upon completion, the script generates a CSV file containing comprehensive information about the position and shape of the detected cells enabling further analysis.

## Prerequisites

Before utilizing the script, ensure the following prerequisites are met:

- Python 3.9.xx

