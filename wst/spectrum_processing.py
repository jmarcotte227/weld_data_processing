'''
Code from: 
https://medium.com/data-science/data-science-for-raman-spectroscopy-a-practical-example-e81c56cf25f
'''
# Loading the required packages:
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.signal.windows import general_gaussian
import sklearn.linear_model as linear_model

# The next function calculates the modified z-scores of a diferentiated spectrum

def modified_z_score(ys):
    ysb = np.diff(ys) # Differentiated intensity values
    median_y = np.median(ysb) # Median of the intensity values
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ysb]) # median_absolute_deviation of the differentiated intensity values
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in ysb] # median_absolute_deviationmodified z scores
    return modified_z_scores
    
# The next function calculates the average values around the point to be replaced.
def fixer(y,ma):
    threshold = 7 # binarization threshold
    spikes = abs(np.array(modified_z_score(y))) > threshold
    y_out = y.copy()
    for i in np.arange(len(spikes)):
        if spikes[i] != 0:
            w = np.arange(i-ma,i+1+ma)
            we = w[spikes[w] == 0]
            y_out[i] = np.mean(y[we])
    return y_out

# Baseline stimation function:
def baseline_als(y, lam, p, niter=100):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

if __name__=='__main__':
    TIME_STEP = 100
    # TEST_ID = 'KEEP_40IPM_braid_120250807-113051'
    TEST_ID = 'Keep_80IPM_bead_220250807-114712'
    data_dir = f'../../wst_data/{TEST_ID}/'

    # load spectrum data
    spectrum_ts = np.loadtxt(f"{data_dir}/spec_counts.csv",
                             delimiter=',',
                             skiprows=1)
    wavelengths_ts = np.loadtxt(f"{data_dir}/spec_wavelengths.csv", 
                                delimiter=',',
                                skiprows=1)

    # process spectrum and wavelengths
    wavelengths = wavelengths_ts[0,1:]
    # spectrum = spectrum_ts[TIME_STEP, 1:]

    # sum spectrum
    spectrum = np.sum(spectrum_ts[:,1:], axis=0)

    
    despiked_spectrum = fixer(spectrum,ma=10)
    # Estimation of the baseline:
    # Parameters for this case:
    l = 10000000 # smoothness
    p = 0.05 # asymmetry
    estimated_baselined = baseline_als(despiked_spectrum, l, p)

    # Baseline subtraction:
    baselined_spectrum = despiked_spectrum - estimated_baselined

    # Smooting Parameters:
    w = 9 # window (number of points)
    p = 2 # polynomial order

    smoothed_spectrum = savgol_filter(baselined_spectrum, w, polyorder = p, deriv=0)

    # plt.plot(wavelengths, spectrum, color = 'black', label = 'Baselined spectrum with noise' )
    # plt.plot(wavelengths, smoothed_spectrum, color = 'red', label = 'Smoothed spectrum' )
    # plt.title('Smoothed spectrum', fontsize = 15)
    plt.plot(wavelengths, spectrum, color = 'blue', label = 'Raw spectrum' )
    plt.title('Raw spectrum', fontsize = 15)
    plt.xlabel('Wavelength (nm)', fontsize = 15)
    plt.ylabel('Intensity (counts)',  fontsize = 15)
    # plt.legend()
    plt.grid()
    plt.show()

    # save data
    save_data = np.vstack((wavelengths.T, smoothed_spectrum.T)).T
    np.savetxt(f"smooth_spec_{TEST_ID}.csv", save_data, delimiter = ',')


