import numpy as np
import cv2
from tqdm import tqdm
from mayavi.mlab import surf 
from mayavi import mlab
import scipy.signal
import numba
from numba import njit, prange
import matplotlib.pyplot as plt


"""
This code is based on this paper:
http://www.sci.utah.edu/~gerig/CS6320-S2015/Materials/Elhabian_SFS08.pdf
"""

class TerrainEstimator:
    eps = 1e-5
    def __init__(self):
        pass

    def estimate_depth(self):
        pass

    def estimate_lighting(self, image):
        normalized = image / np.max(image)
        avg_brightness = np.mean(normalized)
        avg_brightness_sq = np.mean(np.mean(normalized**2.0))
        gradx, grady = np.gradient(normalized)
        grad_vecs = np.sqrt(gradx**2.0 + grady**2.0)
        normalx = gradx / (grad_vecs + self.eps)
        normaly = grady / (grad_vecs + self.eps)
        gradx_avg = np.mean(gradx)
        grady_avg = np.mean(grady)

        gamma = np.sqrt((6 * (np.pi**2.0) * avg_brightness_sq) - (48 * avg_brightness**2.0))
        albedo = gamma / np.pi

        if ((4 * avg_brightness) / gamma) > 1.0:
            slant = 0.00001
        else:
            slant = np.arccos((4 * avg_brightness) / gamma)
        tilt = np.arctan2(gradx_avg, grady_avg)

        if tilt < 0:
            tilt += np.pi

        I = np.array([np.cos(tilt) * np.sin(slant), np.sin(tilt) * np.sin(slant), np.cos(slant)])
        return albedo, slant, tilt, I

    def estimate_surface_fft(self, image, filter_window=1):
        E = image / np.max(image)
        albedo, slant, tilt, I = self.estimate_lighting(image)
        Fe = scipy.fft.fft2(E)
        x, y = np.meshgrid(E.shape[1], E.shape[0])
        wx = (2*np.pi*x) / E.shape[0]
        wy = (2*np.pi*y) / E.shape[1]
        Fz = Fe / (-1j * wx * np.cos(tilt) * np.sin(slant) - 1j * wy * np.sin(tilt) * np.sin(slant))
        Z = np.abs(scipy.fft.ifft2(Fz))
        Z[Z == np.inf] = 0
        Z[Z == np.nan] = 0

        Z_filtered = scipy.signal.medfilt2d(Z, filter_window)
        Z_max = np.percentile(np.abs(Z_filtered), 95)
        Z_filtered[np.abs(Z_filtered) > Z_max] = Z_max
        Z_filtered /= Z_max
        p, q = np.gradient(Z_filtered)
        return Z_filtered, p, q

def full_estimate(image):
    estimator = TerrainEstimator()
    Z, Z_x, Z_y = estimator.estimate_surface_fft(image)
    return variance_window(Z_x, Z_y)

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def variance_window(p, q, window_size=13):
    x_win = rolling_window(p, window_size)
    x_var = np.var(x_win, axis=-1)
    y_win = rolling_window(q, window_size)
    y_var = np.var(y_win, axis=-1)
    summed_var = x_var + y_var
    return summed_var

if __name__ == "__main__":
    image = cv2.imread("crater2.jpg", cv2.IMREAD_GRAYSCALE)
    summed_var = full_estimate(image)
    plt.imshow(summed_var)
    plt.show()
        
        
