# Conditioning

Welcome to the **Conditioning** section of the **Cellular Vibration Analysis using Atomic Force Microscopy (AFM)** project. This directory contains code for conditioning raw data files, making them more accessible and easy to work with in subsequent steps.

## Purpose

The purpose of the 'conditioning' process is to transform raw data obtained from AFM experiments into a structured and readable format, specifically a CSV file. This conditioning step is essential to prepare the data for further analysis, making it accessible for subsequent model fitting codes.

## Content

The 'conditioning' folder includes the following:

- **/src**: This directory contains the source code for data conditioning. The primary script is `conditioning_script.py`, which processes raw AFM data and generates a CSV file.

- **/data**: Sample raw data files used for testing the conditioning script. 

## Getting Started

To use the conditioning script:

1. Run the conditioning script
   python conditioning.py

2. The script will process the raw data files from '/conditioning/data' and generate a CSV file in the same directory.

3. The conditioned data in CSV format is now ready for use by the model fitting codes in other sections of the project.

## Contributing

We welcome contributions to enhance the data conditioning capabilities of our project. If you'd like to contribute, please follow standard contribution guidelines.

## License

This project is developed at the **Instituto de Micro y Nanotecnología**, part of **CSIC: Consejo Superior de Investigaciones Científicas**, within the research group Bionanomechanics. All rights reserved by the respective authors and contributors.

For more details, refer to the information provided by the [Bionanomechanics research group](https://bionano.imn-cnm.csic.es/?lang=en).
