#!/usr/bin/env python
# Compute edge score for given image
# 1. Smooth using L0 gradient minimization method
# https://github.com/kjzhang/kzhang-cs205-l0-smoothing
# 2. Canny
# https://en.wikipedia.org/wiki/Canny_edge_detector

import argparse
from os.path import exists, basename
import cv2
import os
import numpy as np
from os.path import splitext, join, dirname
from matplotlib.image import imread

# Convert point-spread function to optical transfer function
def psf2otf(psf, outSize=None):
    # Prepare psf for conversion
    data = prepare_psf(psf, outSize)

    # Compute the OTF
    otf = np.fft.fftn(data)

    return np.complex64(otf)


def prepare_psf(psf, outSize=None, dtype=None):
    if not dtype:
        dtype = np.float32

    psf = np.float32(psf)

    # Determine PSF / OTF shapes
    psfSize = np.int32(psf.shape)
    if not outSize:
        outSize = psfSize
    outSize = np.int32(outSize)

    # Pad the PSF to outSize
    new_psf = np.zeros(outSize, dtype=dtype)
    new_psf[:psfSize[0], :psfSize[1]] = psf[:, :]
    psf = new_psf

    # Circularly shift the OTF so that PSF center is at (0,0)
    shift = -(psfSize / 2)
    psf = circshift(psf, shift)

    return psf


# Circularly shift array
def circshift(A, shift):
    for i in range(shift.size):
        A = np.roll(A, np.round(shift[i]).astype(np.int32), axis=i)
    return A

# Smooth using L0 gradient minimization
def l0_smoothing(image_path, kappa=2.0, _lambda=2e-2):
    # Read image I
    image = imread(image_path)# cv2.imread(image_path)
    print(image.shape, type(image), image[0].dtype, np.min(image), np.max(image))

    # Validate image format
    assert len(image.shape) == 3, "Error: input must be 3-channel RGB image"

    N, M, D = np.int32(image.shape)
    print("Processing {:d} x {:d} RGB image".format(M, N))

    # Initialize S as I
    S = np.float32(image) / 256

    # Compute image OTF
    size_2D = [N, M]
    fx = np.int32([[1, -1]])
    fy = np.int32([[1], [-1]])
    otfFx = psf2otf(fx, size_2D)
    otfFy = psf2otf(fy, size_2D)

    # Compute F(I)
    FI = np.complex64(np.zeros((N, M, D)))
    FI[:, :, 0] = np.fft.fft2(S[:, :, 0])
    FI[:, :, 1] = np.fft.fft2(S[:, :, 1])
    FI[:, :, 2] = np.fft.fft2(S[:, :, 2])

    # Compute MTF
    MTF = np.power(np.abs(otfFx), 2) + np.power(np.abs(otfFy), 2)
    MTF = np.tile(MTF[:, :, np.newaxis], (1, 1, D))

    # Initialize buffers
    h = np.float32(np.zeros((N, M, D)))
    v = np.float32(np.zeros((N, M, D)))
    dxhp = np.float32(np.zeros((N, M, D)))
    dyvp = np.float32(np.zeros((N, M, D)))
    FS = np.complex64(np.zeros((N, M, D)))

    # Iteration settings
    beta_max = 1e5;
    beta = 2 * _lambda
    iteration = 0

    # Iterate until desired convergence in similarity
    while beta < beta_max:
        print("iteration {:d}".format(iteration))

        ### Step 1: estimate (h, v) subproblem

        # compute dxSp
        h[:, 0:M - 1, :] = np.diff(S, 1, 1)
        h[:, M - 1:M, :] = S[:, 0:1, :] - S[:, M - 1:M, :]

        # compute dySp
        v[0:N - 1, :, :] = np.diff(S, 1, 0)
        v[N - 1:N, :, :] = S[0:1, :, :] - S[N - 1:N, :, :]

        # compute minimum energy E = dxSp^2 + dySp^2 <= _lambda/beta
        t = np.sum(np.power(h, 2) + np.power(v, 2), axis=2) < _lambda / beta
        t = np.tile(t[:, :, np.newaxis], (1, 1, 3))

        # compute piecewise solution for hp, vp
        h[t] = 0
        v[t] = 0

        ### Step 2: estimate S subproblem

        # compute dxhp + dyvp
        dxhp[:, 0:1, :] = h[:, M - 1:M, :] - h[:, 0:1, :]
        dxhp[:, 1:M, :] = -(np.diff(h, 1, 1))
        dyvp[0:1, :, :] = v[N - 1:N, :, :] - v[0:1, :, :]
        dyvp[1:N, :, :] = -(np.diff(v, 1, 0))
        normin = dxhp + dyvp

        FS[:, :, 0] = np.fft.fft2(normin[:, :, 0])
        FS[:, :, 1] = np.fft.fft2(normin[:, :, 1])
        FS[:, :, 2] = np.fft.fft2(normin[:, :, 2])

        # solve for S + 1 in Fourier domain
        denorm = 1 + beta * MTF;
        FS[:, :, :] = (FI + beta * FS) / denorm

        # inverse FFT to compute S + 1
        S[:, :, 0] = np.float32((np.fft.ifft2(FS[:, :, 0])).real)
        S[:, :, 1] = np.float32((np.fft.ifft2(FS[:, :, 1])).real)
        S[:, :, 2] = np.float32((np.fft.ifft2(FS[:, :, 2])).real)

        # update beta for next iteration
        beta *= kappa
        iteration += 1

    # Rescale image
    S = (S - np.min(S)) / np.ptp(S)
    S = S * 255.0

    return S


def get_prefix(file_path):
    return join(dirname(file_path), basename(splitext(file_path)[0]))

def edge_detection(image_path, l0_sth_kappa, l0_sth_lambda, canny_min_val, canny_max_val):
    im = l0_smoothing(image_path, l0_sth_kappa, l0_sth_lambda)

    # https://theailearner.com/tag/non-max-suppression/
    # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    # im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print(im.shape, type(im), im[0].dtype, np.min(im), np.max(im))

    return cv2.Canny(im, canny_min_val, canny_max_val, L2gradient=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute edge of the image using L0 smoothing and Canny edge detector.")
    parser.add_argument("path", help="Path to image", type=str)
    parser.add_argument("--kappa_smooth", help="Kappa used for L0 smoothing", type=float, default=2.0)
    parser.add_argument("--lambda_smooth", help="Lambda used for L0 smoothing", type=float, default=2e-2)
    parser.add_argument("--min_val", help="Canny hysteresis min value", type=float, default=50)
    parser.add_argument("--max_val", help="Canny hysteresis max value", type=float, default=200)
    args = parser.parse_args()

    # img_path = join(dir_path, "images", "pano", "image{:04d}.png".format(readout_idx))
    if not exists(args.path):
        exit(args.path, " could not be found")

    edges = edge_detection(args.path, args.kappa_smooth, args.lambda_smooth, args.min_val, args.max_val)
    print(edges.shape, type(edges), edges[0].dtype, np.min(edges), np.max(edges))

    edges_path = get_prefix(args.path) + "_edges.jpg"
    cv2.imwrite(edges_path, edges)
    print("Edges saved to {}".format(edges_path))

