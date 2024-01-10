# Cell Resonance Analysis

## Introduction

This project delves into the analysis of cell resonances using an Atomic Force Microscope (AFM) in a liquid cell culture environment. The experiment involves measuring the vertical and lateral deflection of a cantilever immersed in cell culture liquid. Spectra are extracted through a Fourier transform applied to both the vertical and lateral deflection signals.

The experimental procedure encompasses acquiring spectra with the cantilever both without and with a cell attached to its tip. Initially, the spectrum is obtained without the cell, and then the cantilever is approached to the cell until attachment occurs. Subsequently, the spectrum is measured again under this condition.

The `conditioning.py` script, available in [this GitHub project](https://github.com/javier-rueda/Cell-Resonances/tree/main/data_analysis/conditioning), plays a pivotal role in processing the raw data obtained from LabView. This script ensures the data is formatted appropriately for the subsequent analysis.

In the fitting script presented here, we employ the Mass Transfer Function to model the resonance behavior of the cell-cantilever system. The Mass Transfer Function is utilized as a theoretical framework to fit our experimental data, enabling the extraction of mechanical and vibrational properties specific to the cell.

This documentation provides insights into the theoretical background, data structure, and the step-by-step procedures employed in the analysis. It serves as a comprehensive resource for understanding the application of the Mass Transfer Function in elucidating the mechanical properties of cells through AFM.
