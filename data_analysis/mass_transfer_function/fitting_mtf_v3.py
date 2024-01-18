

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import tkinter as tk
import os
from tkinter import filedialog
from matplotlib.ticker import ScalarFormatter
from scipy.signal import chirp, find_peaks, peak_widths
from sklearn.metrics import r2_score
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
import re




def browse_csv(prompt):
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        file_path = filedialog.askopenfilename(title=prompt, filetypes=[("CSV Files", "*.csv")])

        if not file_path:
            print("No file selected. Exiting.")
            return None, None

        print("Loading DataFrame, please wait...")
        df = pd.read_csv(file_path, sep=';', decimal=',', index_col=0)
        filename = os.path.basename(file_path)  # Get just the filename
        return df, filename
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None







# Function to find the indices of the two closest numbers with a minimum distance
def find_closest_indices(arr, value, min_distance=5):
    abs_diff = np.abs(arr - value)
    idx_min_diff = np.argmin(abs_diff)
    
    # Exclude the first minimum difference to avoid selecting the same index
    abs_diff[idx_min_diff] = np.inf
    
    # Find the second index with a minimum distance
    idx_second_min_diff = np.argmin(abs_diff)
    
    # Ensure the minimum distance constraint
    while np.abs(idx_second_min_diff - idx_min_diff) < min_distance:
        abs_diff[idx_second_min_diff] = np.inf
        idx_second_min_diff = np.argmin(abs_diff)

    return idx_min_diff, idx_second_min_diff







def peak_identification_torsional(y_data, x_data, width_peaks=3):
    ''' We will define now a function that finds the peaks and valleys,
    as well as the FWHM of the peaks.
    It also gives a first estimation of the value of the Q-factors.
    IMPORTANT: The input must be Array of float64. '''
    
    
    peak_id = pd.DataFrame()
    valley_id = pd.DataFrame()
     
    
    
    # Find the peaks in the data
    peaks, _ = find_peaks(y_data, distance=10, width=width_peaks)
       

    # Get the frequencies and amplitudes of the peaks and valleys
    peak_frequencies = x_data[peaks]
    peak_amplitudes = y_data[peaks]
    peak_id['peak_freq'] = peak_frequencies
    peak_id['peak_amplitude'] = peak_amplitudes
    
    # Calculate FWHM for each identified peak using peak_widths
    fwhm = peak_widths(y_data, peaks, rel_height=0.5)  # rel_height=0.5 corresponds to half maximum
    fwhm_max = peak_widths(y_data, peaks, rel_height=1)
    
    # fwhm_index = fwhm[0]
    fwhm_freq1_index = fwhm[2].astype(int)
    fwhm_freq1 = x_data[fwhm_freq1_index]
    fwhm_freq2_index = fwhm[3].astype(int)
    fwhm_freq2 = x_data[fwhm_freq2_index]
    fwhm_max_freq1_index = fwhm_max[2].astype(int)
    fwhm_max_freq1 = x_data[fwhm_max_freq1_index]
    fwhm_max_freq2_index = fwhm_max[3].astype(int)
    fwhm_max_freq2 = x_data[fwhm_max_freq2_index]
    
    peak_id['peak_freq'] = peak_frequencies
    peak_id['peak_amplitude'] = peak_amplitudes
    peak_id['FWHM_y'] = fwhm[1]
    peak_id['FWHM_x1'] = fwhm_freq1
    peak_id['FWHM_x2'] = fwhm_freq2
    peak_id['FWHM_max_y'] = fwhm_max[1]
    peak_id['FWHM_max_x1'] = fwhm_max_freq1
    peak_id['FWHM_max_x2'] = fwhm_max_freq2
    peak_id['FWHM'] = peak_id['FWHM_x2']-peak_id['FWHM_x1']
    peak_id['Q_factor'] = peak_id['peak_freq']/peak_id['FWHM']
    
 
    
    
    '''
    Since the torsional spectrum has only two real peaks, the 1Torsional and 2Torsional,
    we will be selecting them by choosing the highest peak (1T) and the last peak (2T)
    '''    
    # Eliminate rows where 'peak_freq' > 550000
    filtered_peak_id = peak_id[peak_id['peak_freq'] <= 550000]
    # Get the row with the highest 'peak_freq' from the remaining set
    max_freq_row = filtered_peak_id.nlargest(1, 'peak_freq')
    # Get the two rows with the highest 'peak_amplitude' from the remaining set
    top_two_amplitude_rows = filtered_peak_id.nlargest(2, 'peak_amplitude')
    # Get the row with the highest 'peak_freq' among the top two amplitude rows
    result_row = top_two_amplitude_rows.nlargest(1, 'peak_freq')
    # Concatenate the two selected rows
    final_result = pd.concat([max_freq_row, result_row], ignore_index=True)
    peak_id = final_result
    
    
    ''' We will be selecting the valleys as the first valley detected left to the peak.'''
    valley_id['valley_freq'] = peak_id['FWHM_max_x1']
    valley_id['valley_amplitude'] = peak_id['FWHM_max_y']
    
    
    # ---------------------------------------------------------------------------
    ''' There are two ways of valley detection. The first one above this line
    is the ideal one if peaks are perfectly detected by the program.
    However, it is better to use the method below if the data is noisy. '''
    
    
    
    
    
    idx_peaks = np.where(np.isin(x_data, peak_id['peak_freq']))[0]   # Index of x_data where the peaks are located
    '''We define the valleys as the minimum value on the left to the peak.
    I have tried with looking for the values of the -log(y_data) but the rate of 
    a successful detection is higher with this method.'''
    idx_valleys = [np.argmin(y_data[:peak_index-1]) for peak_index in idx_peaks]
    
    peak_frequencies = x_data[idx_peaks]
    peak_amplitudes = y_data[idx_peaks]
    valley_frequencies = x_data[idx_valleys]
    valley_amplitudes = y_data[idx_valleys]
        
    valley_id['valley_freq'] = valley_frequencies
    valley_id['valley_amplitude'] = valley_amplitudes
        
# =============================================================================
#     
#     ''' The base of the peaks '''
#     peak_id['FWHM_max_y'] = valley_id['valley_amplitude']
#     index_fwhm_max = [np.array(find_closest_indices(y_data, valley_amplitude, min_distance=20)) for valley_amplitude in valley_amplitudes]
#     peak_id.loc[0, 'FWHM_max_x1'] = x_data[index_fwhm_max[0][0]]
#     peak_id.loc[1, 'FWHM_max_x1'] = x_data[index_fwhm_max[1][0]]
#     peak_id.loc[0, 'FWHM_max_x2'] = x_data[index_fwhm_max[0][1]]
#     peak_id.loc[1, 'FWHM_max_x2'] = x_data[index_fwhm_max[1][1]]
#         
#     
#     ''' The FWHM of the peaks '''
#     fwhm_amplitudes = (peak_amplitudes-valley_amplitudes)/2
#     index_fwhm_y = [np.abs(y_data - fwhm_amplitude).argmin() for fwhm_amplitude in fwhm_amplitudes]
#     peak_id['FWHM_y'] = y_data[index_fwhm_y]
#     index_fwhm = [np.array(find_closest_indices(y_data, fwhm_amplitude, min_distance=20)) for fwhm_amplitude in fwhm_amplitudes]
#         
#     peak_id.loc[0, 'FWHM_x1'] = x_data[index_fwhm[0][0]]
#     peak_id.loc[1, 'FWHM_x1'] = x_data[index_fwhm[1][0]]
#     peak_id.loc[0, 'FWHM_x2'] = x_data[index_fwhm[0][1]]
#     peak_id.loc[1, 'FWHM_x2'] = x_data[index_fwhm[1][1]]
#     
# =============================================================================
    
        
    
    return peak_id, valley_id
    
    
    
    
    


def peak_identification_flexural(y_data, x_data, width_peaks=10, add_LastValley_index=80):
    ''' We will define now a function that finds the peaks and valleys,
    as well as the FWHM of the peaks.
    It also gives a first estimation of the value of the Q-factors.
    IMPORTANT: The input must be Array of float64. '''
    
    
    #print('Identifying Peaks & Valleys...')
    
    peak_id = pd.DataFrame()
    valley_id = pd.DataFrame()
     
    
    
    # Find the peaks in the data
    peaks, _ = find_peaks(y_data, distance=100, width=width_peaks)
    
    '''
    # Find the valleys in the data, we must use the -log(y_data)
    zero_indices = [i for i, value in enumerate(y_data) if value == 0]
    for index in zero_indices:
        if y_data[index - 1] != 0:
            y_data[index] = y_data[index - 1]
        else:
            y_data[index] = y_data[index - 2]
    # It just replaces the 0 from the normalization so that the log doesnt give an error.
    
    neg_y_data_log = -np.log10(y_data)
    valleys, _ = find_peaks(-y_data, height=-30, distance=100, width=1)
    valley_frequencies = x_data[valleys]
    valley_amplitudes = y_data[valleys]
    valley_id['valley_freq'] = valley_frequencies
    valley_id['valley_amplitude'] = valley_amplitudes
    '''
    
       
    
    # Get the frequencies and amplitudes of the peaks and valleys
    peak_frequencies = x_data[peaks]
    peak_amplitudes = y_data[peaks]
    peak_id['peak_freq'] = peak_frequencies
    peak_id['peak_amplitude'] = peak_amplitudes
    
    
    # Calculate FWHM for each identified peak using peak_widths
    fwhm = peak_widths(y_data, peaks, rel_height=0.5)  # rel_height=0.5 corresponds to half maximum
    fwhm_max = peak_widths(y_data, peaks, rel_height=1)
    
    # fwhm_index = fwhm[0]
    fwhm_freq1_index = fwhm[2].astype(int)
    fwhm_freq1 = x_data[fwhm_freq1_index]
    fwhm_freq2_index = fwhm[3].astype(int)
    fwhm_freq2 = x_data[fwhm_freq2_index]
    fwhm_max_freq1_index = fwhm_max[2].astype(int)
    fwhm_max_freq1 = x_data[fwhm_max_freq1_index]
    fwhm_max_freq2_index = fwhm_max[3].astype(int)
    fwhm_max_freq2 = x_data[fwhm_max_freq2_index]
    
    peak_id['peak_freq'] = peak_frequencies
    peak_id['peak_amplitude'] = peak_amplitudes
    peak_id['FWHM_y'] = fwhm[1]
    peak_id['FWHM_x1'] = fwhm_freq1
    peak_id['FWHM_x2'] = fwhm_freq2
    peak_id['FWHM_max_y'] = fwhm_max[1]
    peak_id['FWHM_max_x1'] = fwhm_max_freq1
    peak_id['FWHM_max_x2'] = fwhm_max_freq2
    peak_id['FWHM'] = peak_id['FWHM_x2']-peak_id['FWHM_x1']
    peak_id['Q_factor'] = peak_id['peak_freq']/peak_id['FWHM']
    
    
    ''' Filtering of peak detecion'''
    # Define the specific ranges
    range1 = (0, 16000)
    range2 = (60000, 140000)
    range3 = (250000, 400000)
    
    # Create a boolean mask with multiple conditions
    mask = (
        (peak_id['peak_freq'] >= range1[0]) & (peak_id['peak_freq'] <= range1[1]) |
        (peak_id['peak_freq'] >= range2[0]) & (peak_id['peak_freq'] <= range2[1]) |
        (peak_id['peak_freq'] >= range3[0]) & (peak_id['peak_freq'] <= range3[1])
    )
    
    # Apply the mask to filter the DataFrame
    peak_id = peak_id[mask].reset_index(drop=True)
    
    
    
    ''' We will be selecting the valleys as the first valley detected left to the peak.'''
    valley_id['valley_freq'] = peak_id['FWHM_max_x1']
    valley_id['valley_amplitude'] = peak_id['FWHM_max_y']
    
    
    
    
    # the function returns 2 different dataframes, containing all informatiom about the peaks and valleys.
    return peak_id, valley_id




def plot_spectrum_peaks(x_data, y_data, peaks_df, dpi=300, title='Title', mode='flexural'):
    ''' Let us plot everything so far (smoothed data and peak identification)
    to make sure the program is working correctly. '''

    #print('Drawing plot...')
    # Plot the original data and mark the identified peaks
    plt.figure(dpi=dpi)
    plt.plot(x_data, y_data, lw=0.5, label="Original Data")

    # Plot marks on peaks and valleys
    plt.plot(peaks_df['peak_freq'], peaks_df['peak_amplitude'], 'rx', markersize=3, label="Peaks")
    plt.plot(peaks_df['FWHM_max_x1'], peaks_df['FWHM_max_y'], 'bx', markersize=3, label="Valleys")

    
    # Plot lines on FWHM and base of the peak.
    plt.hlines(peaks_df['FWHM_y'], peaks_df['FWHM_x1'], peaks_df['FWHM_x2'],
               lw=0.3, color='green', linestyle='--')
    plt.hlines(peaks_df['FWHM_max_y'], peaks_df['FWHM_max_x1'], peaks_df['FWHM_max_x2'],
               lw=0.3, color='orange', linestyle='--')

    # Annotate peaks with their Q factors
    for i, txt in enumerate(peaks_df['Q_factor']):
        plt.annotate(f'Q={txt:.2f}', (peaks_df['peak_freq'][i], peaks_df['peak_amplitude'][i]),
                     textcoords='offset points', xytext=(10, 5), ha='center', fontsize=6)
    

    # Axis and plot format.
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Amplitude")
    plt.title(title)
    if mode == 'flexural':
        plt.yscale('log')
        plt.ylim(0.01,1.3)
        plt.xlim(1e3, 1e6)
    elif mode == 'torsional':
        plt.yscale('linear')
        plt.ylim(0,1.1)
        plt.xlim(5e3, 1e6)
    plt.xscale('log')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x_data, pos: f'{x_data / 1000:.0f}'))
    plt.legend()
    plt.show()




def SHO_hydro_mode(f, An, fn, Qn, Lambda, n):
    term_nominator = An**2 * (f/fn)**Lambda 
    term_denominator = (f**2/fn**2)**(1 + Lambda)  + Qn**2 * (1-(f**2/fn**2))**2
    term_sho = term_nominator / term_denominator
    return term_sho




def SHO_hydro_flexural(f, A1, f1, Q1, Lambda1, A2, f2, Q2, Lambda2, A3, f3, Q3, Lambda3, fcut, WN):

    model_sum = np.sqrt(SHO_hydro_mode(f, A1, f1, Q1, Lambda1, 1) + \
                SHO_hydro_mode(f, A2, f2, Q2, Lambda2, 2) + \
                SHO_hydro_mode(f, A3, f3, Q3, Lambda3, 3) + \
                    fcut/f + WN)
    return model_sum



def SHO_hydro_torsional(f, A1, f1, Q1, Lambda1, A2, f2, Q2, Lambda2, fcut, WN):

    model_sum = np.sqrt(SHO_hydro_mode(f, A1, f1, Q1, Lambda1, 1) + \
                SHO_hydro_mode(f, A2, f2, Q2, Lambda2, 2) + \
                    fcut/f + WN)
    return model_sum



def regions_F(peaks_data, valley_data):

    ''' Now let us start working on the fitting. 
        We will be dividing our data into regions in order to make the fitting 
        progressive.
        The first regions will be delimited by the FWHM.
        The second regions will be delimited by the base of the peaks.
        The last regions will be delimited by the data between valleys. '''
        
    #print("Creating Fitting Regions, please wait...")
    index_regions = ['fwhm', 'base', 'full_peaks']
    columns_regions = ['F1_start', 'F1_end', 'F2_start', 'F2_end', 'F3_start', 'F3_end']
    fitting_regions = pd.DataFrame(index=index_regions, columns=columns_regions)

    fitting_regions.loc['fwhm', 'F1_start'] = peaks_data['FWHM_x1'].values[0]
    fitting_regions.loc['fwhm', 'F1_end'] = peaks_data['FWHM_x2'].values[0]
    fitting_regions.loc['fwhm', 'F2_start'] = peaks_data['FWHM_x1'].values[1]
    fitting_regions.loc['fwhm', 'F2_end'] = peaks_data['FWHM_x2'].values[1]
    fitting_regions.loc['fwhm', 'F3_start'] = peaks_data['FWHM_x1'].values[2]
    fitting_regions.loc['fwhm', 'F3_end'] = peaks_data['FWHM_x2'].values[2]

    fitting_regions.loc['base', 'F1_start'] = peaks_data['FWHM_max_x1'].values[0]
    fitting_regions.loc['base', 'F1_end'] = peaks_data['FWHM_max_x2'].values[0]
    fitting_regions.loc['base', 'F2_start'] = peaks_data['FWHM_max_x1'].values[1]
    fitting_regions.loc['base', 'F2_end'] = peaks_data['FWHM_max_x2'].values[1]
    fitting_regions.loc['base', 'F3_start'] = peaks_data['FWHM_max_x1'].values[2]
    fitting_regions.loc['base', 'F3_end'] = peaks_data['FWHM_max_x2'].values[2]

    fitting_regions.loc['full_peaks', 'F1_start'] = valley_data['valley_freq'].values[0]
    fitting_regions.loc['full_peaks', 'F1_end'] = valley_data['valley_freq'].values[1]
    fitting_regions.loc['full_peaks', 'F2_start'] = valley_data['valley_freq'].values[1]
    fitting_regions.loc['full_peaks', 'F2_end'] = valley_data['valley_freq'].values[2]
    fitting_regions.loc['full_peaks', 'F3_start'] = valley_data['valley_freq'].values[2]
    fitting_regions.loc['full_peaks', 'F3_end'] = 500000

    fitting_regions = fitting_regions.astype(float)
    return fitting_regions



def regions_T(peaks_data, valley_data):
    
    ''' Now let us start working on the fitting. 
        We will be dividing our data into regions in order to make the fitting 
        progressive.
        The first regions will be delimited by the FWHM.
        The second regions will be delimited by the base of the peaks.
        The last regions will be delimited by the data between valleys. '''
        
    #print("Creating Fitting Regions, please wait...")
    index_regions = ['fwhm', 'base', 'full_peaks']
    columns_regions = ['F1_start', 'F1_end', 'F2_start', 'F2_end', 'F3_start', 'F3_end']
    fitting_regions = pd.DataFrame(index=index_regions, columns=columns_regions)

    fitting_regions.loc['fwhm', 'F1_start'] = peaks_data['FWHM_x1'].values[0]
    fitting_regions.loc['fwhm', 'F1_end'] = peaks_data['FWHM_x2'].values[0]
    fitting_regions.loc['fwhm', 'F2_start'] = peaks_data['FWHM_x1'].values[1]
    fitting_regions.loc['fwhm', 'F2_end'] = peaks_data['FWHM_x2'].values[1]

    fitting_regions.loc['base', 'F1_start'] = peaks_data['FWHM_max_x1'].values[0]
    fitting_regions.loc['base', 'F1_end'] = peaks_data['FWHM_max_x2'].values[0]
    fitting_regions.loc['base', 'F2_start'] = peaks_data['FWHM_max_x1'].values[1]
    fitting_regions.loc['base', 'F2_end'] = peaks_data['FWHM_max_x2'].values[1]

    fitting_regions.loc['full_peaks', 'F1_start'] = 50000 #valley_data['valley_freq'].values[0]
    fitting_regions.loc['full_peaks', 'F1_end'] = valley_data['valley_freq'].values[1]
    fitting_regions.loc['full_peaks', 'F2_start'] = valley_data['valley_freq'].values[1]
    fitting_regions.loc['full_peaks', 'F2_end'] = 700000 #peaks_data['FWHM_max_x2'].values[1]

    fitting_regions = fitting_regions.astype(float)
    return fitting_regions



def range_fit(fitting_regions, x_data, y_data, flat_i=50, start_mode=1, end_mode=3):

    start_value = fitting_regions.loc['full_peaks', f'F{start_mode}_start']
    end_value = fitting_regions.loc['full_peaks', f'F{end_mode}_end']
    
    start_index = np.argmin(np.abs(x_data - start_value))
    end_index = np.argmin(np.abs(x_data - end_value))

    ''' Get the data that for the fitting. '''
    # This factor reduces the number of points in the fitting range from both sides.
    x_values = x_data[start_index+flat_i : int(end_index-flat_i)]
    y_values = y_data[start_index+flat_i : int(end_index-flat_i)]
    
    return x_values, y_values




def initial_values_F(peaks):
    A1_initial = peaks.loc[0, 'peak_amplitude']
    A2_initial = peaks.loc[1, 'peak_amplitude']
    A3_initial = peaks.loc[2, 'peak_amplitude']
    
    f1_initial = peaks.loc[0, 'peak_freq']
    f2_initial = peaks.loc[1, 'peak_freq']
    f3_initial = peaks.loc[2, 'peak_freq']
    
    Q1_initial = peaks.loc[0, 'Q_factor']
    Q2_initial = peaks.loc[1, 'Q_factor']
    Q3_initial = peaks.loc[2, 'Q_factor']
    
    Lambda1_initial = 0
    Lambda2_initial = 0.5
    Lambda3_initial = 0.7
    
    fcut_initial = 0.1
    WN_initial = 0.1
    
    initial_params = [A1_initial, f1_initial, Q1_initial, Lambda1_initial,
                     A2_initial, f2_initial, Q2_initial, Lambda2_initial,
                     A3_initial, f3_initial, Q3_initial, Lambda3_initial,
                     fcut_initial, WN_initial]
    
    return initial_params



def initial_values_T(peaks):
    A1_initial = peaks.loc[0, 'peak_amplitude']
    A2_initial = peaks.loc[1, 'peak_amplitude']
    
    f1_initial = peaks.loc[0, 'peak_freq']
    f2_initial = peaks.loc[1, 'peak_freq']
    
    Q1_initial = peaks.loc[0, 'Q_factor']
    Q2_initial = peaks.loc[1, 'Q_factor']
    
    Lambda1_initial = 0
    Lambda2_initial = 0.5
    
    fcut_initial = 0.1
    WN_initial = 0.1
    
    initial_params = [A1_initial, f1_initial, Q1_initial, Lambda1_initial,
                     A2_initial, f2_initial, Q2_initial, Lambda2_initial,
                     fcut_initial, WN_initial]
    
    return initial_params




def create_parametersDf(mode):
    if mode == 'flexural':
        index_values = ["Amplitude 1F", "Frequency 1F", "QFactor 1F", "Lambda 1F",
                        "Amplitude 2F", "Frequency 2F", "QFactor 2F", "Lambda 2F",
                        "Amplitude 3F", "Frequency 3F", "QFactor 3F", "Lambda 3F",
                        "WN", "Fcut",  
                        "R2", "MSE"]
    elif mode == 'torsional': 
        index_values = ["Amplitude 1T", "Frequency 1T", "QFactor 1T", "Lambda 1T",
                        "Amplitude 2T", "Frequency 2T", "QFactor 2T", "Lambda 2T",
                        "WN", "Fcut",  
                        "R2", "MSE"]
    df = pd.DataFrame(index=index_values)
    return df




def bounds_SHO_F(peaks, min_resFactor=0.9, max_resFactor=1.3, min_QFactor=0.1, max_QFactor=10):
    A1_min = min_resFactor* peaks.loc[0, 'peak_amplitude']
    f1_min = min_resFactor* peaks.loc[0, 'peak_freq']
    Q1_min = min_QFactor * peaks.loc[0, 'Q_factor']
    Lambda1_min = 0
               
    A2_min = min_resFactor* peaks.loc[1, 'peak_amplitude']
    f2_min = min_resFactor* peaks.loc[1, 'peak_freq']
    Q2_min = min_QFactor * peaks.loc[1, 'Q_factor']
    Lambda2_min = 0.3
               
    A3_min = min_resFactor* peaks.loc[2, 'peak_amplitude']
    f3_min = min_resFactor* peaks.loc[2, 'peak_freq']
    Q3_min = min_QFactor * peaks.loc[2, 'Q_factor']
    Lambda3_min = 0.6
              
    fcut_min = 0
    WN_min = 0
    
    # --------------

    A1_max = 1
    f1_max = max_resFactor*  peaks.loc[0, 'peak_freq']
    Q1_max = max_QFactor * peaks.loc[0, 'Q_factor']
    Lamda1_max = 1
               
    A2_max = max_resFactor* peaks.loc[1, 'peak_amplitude']
    f2_max = max_resFactor* peaks.loc[1, 'peak_freq']
    Q2_max = max_QFactor * peaks.loc[1, 'Q_factor']
    Lambda2_max = 1
               
    A3_max = max_resFactor* peaks.loc[2, 'peak_amplitude']
    f3_max = max_resFactor* peaks.loc[2, 'peak_freq']
    Q3_max = max_QFactor * peaks.loc[2, 'Q_factor']
    Lambda3_max = 1
    
    fcut_max = 1
    WN_max = 1
    
    
    bounds = ([A1_min, f1_min, Q1_min, Lambda1_min,
               A2_min, f2_min, Q2_min, Lambda2_min,
               A3_min, f3_min, Q3_min, Lambda3_min,
               fcut_min, WN_min],
              [A1_max, f1_max, Q1_max, Lamda1_max,
               A2_max, f2_max, Q2_max, Lambda2_max,
               A3_max, f3_max, Q3_max, Lambda3_max,
               fcut_max, WN_max])
    return bounds
    



def bounds_SHO_T(peaks, min_resFactor=0.9, max_resFactor=1.3, min_QFactor=0.1, max_QFactor=10):
    A1_min = min_resFactor* peaks.loc[0, 'peak_amplitude']
    f1_min = min_resFactor* peaks.loc[0, 'peak_freq']
    Q1_min = min_QFactor * peaks.loc[0, 'Q_factor']
    Lambda1_min = 0
               
    A2_min = min_resFactor* peaks.loc[1, 'peak_amplitude']
    f2_min = min_resFactor* peaks.loc[1, 'peak_freq']
    Q2_min = min_QFactor * peaks.loc[1, 'Q_factor']
    Lambda2_min = 0
              
    fcut_min = 0
    WN_min = 0
    
    # --------------

    A1_max = 1
    f1_max = max_resFactor*  peaks.loc[0, 'peak_freq']
    Q1_max = max_QFactor * peaks.loc[0, 'Q_factor']
    Lamda1_max = 1
               
    A2_max = max_resFactor* peaks.loc[1, 'peak_amplitude']
    f2_max = max_resFactor* peaks.loc[1, 'peak_freq']
    Q2_max = max_QFactor * peaks.loc[1, 'Q_factor']
    Lambda2_max = 1
               
    
    fcut_max = 1
    WN_max = 1
    
    
    bounds = ([A1_min, f1_min, Q1_min, Lambda1_min,
               A2_min, f2_min, Q2_min, Lambda2_min,
               fcut_min, WN_min],
              [A1_max, f1_max, Q1_max, Lamda1_max,
               A2_max, f2_max, Q2_max, Lambda2_max,                
               fcut_max, WN_max])
    return bounds
    












def plot_fitting(x_data, y_data, x_fit, y_fit, params, title, mode='flexural'):
    ''' Let us plot everything so far (smoothed data and peak identification)
    to make sure the program is working correctly. '''

    # Plot the original data and mark the identified peaks
    plt.figure(dpi=1200)
    plt.plot(x_data, y_data, lw=5, alpha=0.3, label="Original Data" , color='gray')
    plt.plot(x_fit, y_fit, lw=0.7, label="Fitting", linestyle='--', color='black', alpha=0.7)
    if mode == 'flexural':
        plt.scatter(params[1],params[0], s=5)
        plt.scatter(params[5],params[4], s=5)
        plt.scatter(params[9],params[8], s=5)
        plt.yscale('log')
        plt.ylim(0.01,1.3)
        plt.xlim(1e3, 1e6)
    elif mode == 'torsional':
        plt.scatter(params[1],params[0], s=5)
        plt.scatter(params[5],params[4], s=5)
        plt.yscale('linear')
        plt.ylim(0.01,1.1)
        plt.xlim(5e3, 1e6)
    
    # Axis and plot format.
    plt.title(title)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Amplitude")
    plt.xscale('log')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x_data, pos: f'{x_data / 1000:.0f}'))
    plt.legend()
    plt.show()
    

 

def iterative_fitting(params, parameter_bounds, xFit_initial, yFit_initial, mode='flexural',
                      N=100000, r2_threshold=0.9999, param_change_threshold=0.01):
    #print("Starting Iterative Fitting...")
    covariance = None
    for i in range(N):
        if mode == 'flexural':
            yFit = SHO_hydro_flexural(xFit_initial, params[0], params[1], params[2], params[3],
                                  params[4], params[5], params[6], params[7], params[8],
                                  params[9], params[10], params[11], params[12], params[13])
        elif mode == 'torsional':
            yFit = SHO_hydro_torsional(xFit_initial, params[0], params[1], params[2], params[3],
                                  params[4], params[5], params[6], params[7], params[8],
                                  params[9])
        r2 = r2_score(yFit_initial, yFit)
        # mse = mean_squared_error(yFit_initial, yFit)
        if r2 >= r2_threshold:
            break
        try:
            if mode == 'flexural':
                new_params, covariance = curve_fit(SHO_hydro_flexural, xFit_initial, yFit_initial, method='trf',
                                                   p0=params, bounds=parameter_bounds, maxfev=100000)
            elif mode == 'torsional':
                
                new_params, covariance = curve_fit(SHO_hydro_torsional, xFit_initial, yFit_initial, method='trf',
                                                   p0=params, bounds=parameter_bounds, maxfev=100000)
        except RuntimeError:
            print("Curve fit did not converge. Exiting.")
            break

        if np.max(np.abs(new_params - params) / np.abs(params)) < param_change_threshold:
            break

        params = new_params


    #print("Iterative Fitting Completed!")

    return params, covariance




def clean_dataframe(fitting_parameters, threshold=3):
    """
    Clean a DataFrame by removing columns with outlier parameter values, excluding the first column.

    Parameters:
    - fitting_parameters (pd.DataFrame): Input DataFrame with parameters.
    - threshold (float): Z-score threshold for considering a value as an outlier. Default is 3.

    Returns:
    - pd.DataFrame: Cleaned DataFrame with outliers removed.
    """
    
    
    # Drops the columns that has NaN values (columns that weren't correctly fit.)
    fitting_parameters = fitting_parameters.dropna(axis=1)    
    
    # Save the first column (initial conditions)
    initial_conditions = fitting_parameters.iloc[:, 0]

    # Exclude the first column for outlier detection
    data_for_outlier_detection = fitting_parameters.iloc[:, 1:]

    # Calculate z-scores for each column excluding the first
    z_scores = np.abs(zscore(data_for_outlier_detection.T))
    z_scores = z_scores.T

    # Identify columns with outliers
    outliers = (z_scores > threshold).any(axis=0)
    
    outliers = outliers.reset_index(drop=True)
    outliers.name = 'Outliers'
    outliers.index = data_for_outlier_detection.columns
    
    data_for_outlier_detection = data_for_outlier_detection.append(outliers)
    # Assuming 'Outliers' is a row in the DataFrame
    outliers_row = data_for_outlier_detection.loc['Outliers']
    
    # Identify columns with 1 in the 'Outliers' row
    columns_to_remove = outliers_row[outliers_row == 1].index
    
    # Remove identified columns from the DataFrame
    cleaned_df = data_for_outlier_detection.drop(columns=columns_to_remove)
    cleaned_df = cleaned_df.drop('Outliers', axis=0)
    cleaned_df.insert(0, initial_conditions.name, initial_conditions)
    
    return cleaned_df
    




def add_time(fitting_parameters):
    time = pd.Series([int(re.search(r'\d+', col).group()) for col in fitting_parameters.columns], 
                         name='Time', index=fitting_parameters.columns)
    fitting_parameters = fitting_parameters.append(time)
    return fitting_parameters





def compute_relative_change(dataframe, columns):
    """
    Compute the relative change for specified columns and add them to the DataFrame.

    Parameters:
    - dataframe: DataFrame containing the columns
    - columns: List of columns for which relative change needs to be computed
    """
    # Iterate through each column
    for column in columns:
        # Compute relative change for the column
        relative_change_column = (dataframe[column] - dataframe[column].iloc[0]) / dataframe[column].iloc[0]

        # Add the relative change column to the DataFrame
        relative_change_column_name = f'{column} RelativeChange'
        dataframe[relative_change_column_name] = relative_change_column
        
    return dataframe



def plot_relative_change(dataframeF, dataframeT, initial_time=0, modesF=3, modesT=2):
    total_num_modes = modesF + modesT
    fig, axes = plt.subplots(nrows=1, ncols=total_num_modes, figsize=(5*total_num_modes, 6), sharey=True, gridspec_kw={'wspace': 0})
    plt.suptitle('Frequency and Q-Factor Relative Change')
    
    
    
    for i in range(total_num_modes):
         
        if i < (total_num_modes-modesT):
            axes[i].plot(dataframeF.index[1:], dataframeF[f'Frequency {i+1}F RelativeChange'][1:], label='Frequency', linewidth=2)
            axes[i].plot(dataframeF.index[1:], dataframeF[f'QFactor {i+1}F RelativeChange'][1:], label='QFactor', linewidth=2) 
            axes[i].axhline(y=0, color='black', linestyle='--', label='Zero Line', linewidth=1)
            axes[i].set_title(f'{i+1}F')
            axes[i].set_xlabel('Time')
            if i == 0:
                axes[i].set_ylabel('Relative Change')
                
        elif i >= (total_num_modes-modesT):
            axes[i].plot(dataframeT.index[1:], dataframeT[f'Frequency {i+1-modesF}T RelativeChange'][1:], label='Frequency', linewidth=2)
            axes[i].plot(dataframeT.index[1:], dataframeT[f'QFactor {i+1-modesF}T RelativeChange'][1:], label='QFactor', linewidth=2) 
            axes[i].axhline(y=0, color='black', linestyle='--', label='Zero Line', linewidth=1)
            axes[i].set_xlabel('Time')
            axes[i].set_title(f'{i+1-modesF}T')
    plt.ylim(-1,0.2)
    plt.legend()
    plt.show()
        
    


#%% MAIN PROGRAM


if __name__=='__main__':
    try:
        
        """ MAIN PROGRAM """
        print(" --------------------- ")
        
        data, filename = browse_csv("Please select the CSV 'conditioned' data file. ")
        fitting_parameters_F = create_parametersDf(mode='flexural')
        fitting_parameters_T = create_parametersDf(mode='torsional')
        
        
        lastSpectrum = int(data.columns[-1].split("_")[-1])    # Obtain the last spectrum number
        for i in range(lastSpectrum):       
            try: 
                # Cycle through all spectrums
                #i=24
                x_frequencies = data['frequencies'].values
                y_flexural = data[f'flexural_{i}'].values
                y_torsional = data[f'torsional_{i}'].values
                
                
                

                '''Identify the peaks and valleys of the spectrum.'''
                peaksF, valleysF = peak_identification_flexural(y_flexural, x_frequencies,
                                                                width_peaks=10)
                peaksT, valleysT = peak_identification_torsional(y_torsional, x_frequencies,
                                                                 width_peaks=0)
                
                
                
                
                '''Plot the spectrum with peaks/valleys.'''
                plot_spectrum_peaks(x_frequencies, y_flexural, peaksF,  title=f"Spectrum Peaks t={i} m", mode='flexural')
                plot_spectrum_peaks(x_frequencies, y_torsional, peaksT,  title=f"Spectrum Peaks t={i} m", mode='torsional')
                
               
                
            
                
                ''' Fit to SHO with Hydrodynamic Factor. 
                    Here we select the data that we will fit,
                    the initial values of the fitting parameters
                    and the bounds of such parameters when performing the fitting.'''
                fitting_regions_F = regions_F(peaksF, valleysF)
                fitting_regions_T = regions_T(peaksT, valleysT)
                
                xFit_flexural, yFitRange_flexural = range_fit(fitting_regions_F, x_frequencies, y_flexural,
                                                              start_mode=1, end_mode=3, flat_i=0)
                xFit_torsional, yFitRange_torsional = range_fit(fitting_regions_T, x_frequencies, y_torsional,
                                                                start_mode=1, end_mode=2, flat_i=0)
                
                initial_guess_F = initial_values_F(peaksF)
                parameter_bounds_F = bounds_SHO_F(peaksF, 
                                              min_resFactor=0.9, max_resFactor=1.5, 
                                              min_QFactor=0.3, max_QFactor=2) 
                
                initial_guess_T = initial_values_T(peaksT)
                parameter_bounds_T = bounds_SHO_T(peaksT, 
                                              min_resFactor=0.7, max_resFactor=2, 
                                              min_QFactor=0.001, max_QFactor=2) 
                
               
                
                
                
                
                try:
                    "Initialize Parameters of the Fitting"
                    params_F, covariance_F = curve_fit(SHO_hydro_flexural, xFit_flexural, yFitRange_flexural, method='trf',
                                                   p0=initial_guess_F, bounds=parameter_bounds_F, maxfev=10000)
                    
                    " Iterative Fitting: It stops at R=0.999 or when the variation in the parameters is <1%"
                    params_F, covariance_F = iterative_fitting(params_F, parameter_bounds_F, 
                                                               xFit_flexural, yFitRange_flexural, 
                                                               mode='flexural', N=1000)
                    yFit_flexural = SHO_hydro_flexural(xFit_flexural, params_F[0], params_F[1], params_F[2], params_F[3],
                                          params_F[4], params_F[5], params_F[6], params_F[7], params_F[8],
                                          params_F[9], params_F[10], params_F[11], params_F[12], params_F[13])
                    
                    ''' Plot the data with the fitting '''
                    plot_fitting(x_frequencies, y_flexural, xFit_flexural, yFit_flexural, params_F, 
                                 title=f"Flexural {i}", mode='flexural')
                    
                    ''' Save Fitting Parameters '''
                    r2_f = r2_score(yFitRange_flexural, yFit_flexural)
                    mse_f = mean_squared_error(yFitRange_flexural, yFit_flexural)   
                    params_F = np.concatenate((params_F, [r2_f, mse_f])) 
                    fitting_parameters_F[f'flexural_{i}'] = params_F
                except Exception as e: print(f'Fitting of flexural {i} does not converge: {e}')
                
                
                
                
                
                # ----------------------------------
                ''' Same with torsional '''
                try:
                    params_T, covariance_T = curve_fit(SHO_hydro_torsional, xFit_torsional, yFitRange_torsional, method='trf',
                                                       p0=initial_guess_T, bounds=parameter_bounds_T, maxfev=10000)
                    params_T, covariance_T = iterative_fitting(params_T, parameter_bounds_T, 
                                                               xFit_torsional, yFitRange_torsional, 
                                                               mode='torsional', N=1000)
                    
                    yFit_torsional = SHO_hydro_torsional(xFit_torsional, params_T[0], params_T[1], params_T[2], params_T[3],
                                          params_T[4], params_T[5], params_T[6], params_T[7], params_T[8], params_T[9])
                    
                    plot_fitting(x_frequencies, y_torsional, xFit_torsional, yFit_torsional, params_T, 
                                 title=f"Torsional {i}", mode='torsional')
                    
                    r2_t = r2_score(yFitRange_torsional, yFit_torsional)
                    mse_t = mean_squared_error(yFitRange_torsional, yFit_torsional)   
                    params_T = np.concatenate((params_T, [r2_t, mse_t])) 
                    fitting_parameters_T[f'torsional_{i}'] = params_T
                except Exception as err: print(f'Fitting of torsional {err} does not converge')
                
                
                
                
            except Exception as e:
                print(f"An error ocurred at iteration {i}: {e}")
        
        
        ''' Add a column with the time stamps. '''
        fitting_parameters_F = add_time(fitting_parameters_F)
        fitting_parameters_T = add_time(fitting_parameters_T)
        
        '''Cleaning Parameter Outliers '''        
        fitting_parameters_F_clean = clean_dataframe(fitting_parameters_F)
        fitting_parameters_T_clean = clean_dataframe(fitting_parameters_T)
        
        
        ''' Transpose the dataframe such that we have each parameter as a column.'''
# =============================================================================
#           IF YOU DO NOT WANT TO CLEAN THE OUTLIERS OF THE DATAFRAME
#         parametersF = fitting_parameters_F.T
#         parametersF.set_index('Time', inplace=True)
#         parametersT = fitting_parameters_T.T
#         parametersT.set_index('Time', inplace=True)
# =============================================================================
        
        parametersF = fitting_parameters_F_clean.T
        parametersF.set_index('Time', inplace=True)
        parametersT = fitting_parameters_T_clean.T
        parametersT.set_index('Time', inplace=True)
        
        
        
        ''' Relative Change in Resonance Frequency & Quality Factor'''
        parametersF = compute_relative_change(parametersF, columns=['Frequency 1F', 'QFactor 1F',
                                                                    'Frequency 2F', 'QFactor 2F',
                                                                    'Frequency 3F', 'QFactor 3F'])
        
        
        parametersT = compute_relative_change(parametersT, columns=['Frequency 1T', 'QFactor 1T',
                                                                    'Frequency 2T', 'QFactor 2T'])
        
        plot_relative_change(parametersF, parametersT)
        
        
        
        
        
        
    except Exception as e:
        print(f"An error ocurred: {e}")
    