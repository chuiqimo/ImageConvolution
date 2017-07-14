from scipy.misc import imread
from scipy.misc import imsave
from scipy import signal
import numpy as np
from Convolution import OneConvolution
import argparse
from scipy.ndimage.filters import gaussian_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=['1', '2', '3', '4'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1':
        #x = np.array([0, 1, 1, 2, 3, 5, 8, 13])
        #w = np.array([0, 0, 1])
        x = np.array([1, 2, 3])
        w = np.array([0,1,0.5])
        model = OneConvolution(x)
        Z = model.convolution(w)
        print(Z)

    if question == '2':
        X = imread('Gothic.jpg', flatten=True)
        imsave('grey.jpg', X)
        W = np.array([[1,2,0]
        	         ,[2,0,-2]
        	         ,[0,-2,-1]])
        grad = signal.convolve2d(X, W)
        imsave('outfile.jpg', grad)

    if question == '3':
        X = imread('Gothic.jpg')
        N,D,r = X.shape
        Z = np.zeros((N,D,r))
        Z = X
        X1 = X[:,:,0]
        X2 = X[:,:,1]
        X3 = X[:,:,2]
        b1 = gaussian_filter(X1, sigma=2)
        b2 = gaussian_filter(X2, sigma=7)
        b3 = gaussian_filter(X3, sigma=2)

        Z[:,:,0] = b1

        imsave('Gaussian1.jpg', Z)

        Z[:,:,1] = b2
        imsave('Gaussian2.jpg', Z)

        Z[:,:,2] = b3
        imsave('Gaussian.jpg', Z)

    if question == '4':
        X = imread('Gothic.jpg')