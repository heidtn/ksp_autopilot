import numpy as np
import cv2
from tqdm import tqdm
from mayavi.mlab import surf 
from mayavi import mlab
import scipy.signal
import numba
from numba import njit, prange



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
            slant = 0.0
        else:
            slant = np.arccos((4 * avg_brightness) / gamma)
        tilt = np.arctan2(gradx_avg, grady_avg)

        if tilt < 0:
            tilt += np.pi

        I = np.array([np.cos(tilt) * np.sin(slant), np.sin(tilt) * np.sin(slant), np.cos(slant)])
        return albedo, slant, tilt, I

    @njit(nopython=True, parallel=True)
    def estimate_surface(self, image, iterations=200):
        albedo, slant, tilt, I = self.estimate_lighting(image)
        print("Got lighting params: ", albedo, slant, tilt, I)
        M, N = image.shape
        p = np.zeros((M, N))
        q = np.zeros((M, N))
        Z = np.zeros((M, N))
        Z_x = np.zeros((M, N))
        Z_y = np.zeros((M, N))
        ix = np.cos(tilt) * np.tan(slant)
        iy = np.sin(tilt) * np.tan(slant)

        for i in prange(iterations):
            R = (np.cos(slant) + p * np.cos(tilt)*np.sin(slant) + q * np.sin(tilt)*np.sin(slant)) /  \
                 np.sqrt(1 + p**2.0 + q**2.0)
            R[R < 0] = 0
            f = image - R
            df_dZ = (p + q) * (ix * p + iy * q + 1) / (np.sqrt((1 + p**2.0 + q**2.0)**3.0)*np.sqrt(1 + ix**2.0 + iy**2.0)) - (ix + iy) / (np.sqrt(1 + p**2.0 + q**2.0)*np.sqrt(1 + ix**2.0 + iy**2.0))
            Z = Z - f / (df_dZ + self.eps)
            Z_x[1:M,:] = Z[0:M-1, :]
            Z_y[:,1:N] = Z[:, 0:N-1]
 
        Z /= 10000000.0
        Z_filtered = scipy.signal.medfilt2d(Z, 17)
        #mlab.surf(Z_filtered)
        #mlab.show()
        return Z, Z_x, Z_y


#@njit(parallel=True)
def fast_estimation(image, albedo, slant, tilt, I, iterations=200):
    eps = 1e-5
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
        Z = Z - f / (df_dZ + eps)
        Z_x[1:M,:] = Z[0:M-1, :]
        Z_y[:,1:N] = Z[:, 0:N-1]

    Z /= 10000000.0
    Z_filtered = scipy.signal.medfilt2d(Z, 17)
    return Z, Z_x, Z_y

def full_estimate(image):
    estimator = TerrainEstimator()
    albedo, slant, tilt, I = estimator.estimate_lighting(image)
    #estimator.estimate_surface(image)
    fast_estimation(image, albedo, slant, tilt, I)


if __name__ == "__main__":
    image = cv2.imread("test_shot.png", cv2.IMREAD_GRAYSCALE)
    full_estimate(image)
        
        
