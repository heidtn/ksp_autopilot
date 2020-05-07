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

    def estimate_surface(self, image, iterations=50, filter_window=13):
        albedo, slant, tilt, I = self.estimate_lighting(image)
        M, N = image.shape
        p = np.zeros((M, N))
        q = np.zeros((M, N))
        Z = np.zeros((M, N))
        Z_x = np.zeros((M, N))
        Z_y = np.zeros((M, N))
        ix = np.cos(tilt) * np.tan(slant)
        iy = np.sin(tilt) * np.tan(slant)

        for i in tqdm(range(iterations)):
            R = (np.cos(slant) + p * np.cos(tilt)*np.sin(slant) + q * np.sin(tilt)*np.sin(slant)) /  \
                 np.sqrt(1 + p**2.0 + q**2.0)
            R[R < 0] = 0
            f = image - R
            df_dZ = (p + q) * (ix * p + iy * q + 1) / (np.sqrt((1 + p**2.0 + q**2.0)**3.0)*np.sqrt(1 + ix**2.0 + iy**2.0)) - (ix + iy) / (np.sqrt(1 + p**2.0 + q**2.0)*np.sqrt(1 + ix**2.0 + iy**2.0))
            Z = Z - f / (df_dZ + self.eps)
            Z_x[1:M,:] = Z[0:M-1, :]
            Z_y[:,1:N] = Z[:, 0:N-1]

            p = Z - Z_x
            q = Z - Z_y

        Z_filtered = scipy.signal.medfilt2d(Z, filter_window)
        Z_filtered[np.isinf(Z_filtered)] = np.nan
        Z_filtered /= np.max(np.abs(Z_filtered))

        s = surf(Z_filtered)
        mlab.show()
        return Z_filtered, p, q

    def estimate_surface_fft(self, image, filter_window=13):
        E = image / np.max(image)
        albedo, slant, tilt, I = self.estimate_lighting(image)
        print(tilt, slant)
        Fe = scipy.fft.fft2(E)
        x, y = np.meshgrid(E.shape[1], E.shape[0])
        wx = (2*np.pi*x) / E.shape[0]
        wy = (2*np.pi*y) / E.shape[1]
        Fz = Fe / (-1j * wx * np.cos(tilt) * np.sin(slant) - 1j * wy * np.sin(tilt) * np.sin(slant))
        Z = np.abs(scipy.fft.ifft2(Fz))
        Z_filtered = scipy.signal.medfilt2d(Z, filter_window)
        Z_filtered /= np.percentile(np.abs(Z_filtered), 0.1)
        s = surf(Z_filtered)

        mlab.show()
        return Z

def full_estimate(image):
    estimator = TerrainEstimator()
    Z, Z_x, Z_y = estimator.estimate_surface_fft(image)
    return variance_window(Z_x, Z_y)

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def variance_window(Z_x, Z_y, window_size=13):
    x_win = rolling_window(Z_x, window_size)
    x_var = np.var(x_win, axis=-1)
    x_var /= np.max(x_var)
    y_win = rolling_window(Z_y, window_size)
    y_var = np.var(y_win, axis=-1)
    y_var /= np.max(y_var)
    summed_var = np.log(x_var + y_var)
    return summed_var

if __name__ == "__main__":
    image = cv2.imread("crater2.jpg", cv2.IMREAD_GRAYSCALE)
    summed_var = full_estimate(image)


    plt.imshow(summed_var)
    plt.show()
        
        
