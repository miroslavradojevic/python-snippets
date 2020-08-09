#!/usr/bin/env python
# Compute edge score from given image

import argparse
from os.path import exists, basename
from matplotlib.image import imread
import cv2
import numpy as np


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
    # print(type(shift.size), shift.size)
    for i in range(shift.size):
        # print(i, type(i))
        # print(type(shift), len(shift), shift.shape)
        # print(type(A), len(A), A.shape)
        # print(shift[i])
        A = np.roll(A, np.round(shift[i]).astype(np.int32), axis=i)
    return A


def l0_smoothing(image_r, kappa=2.0, _lambda=2e-2):
    # Read image I
    image = cv2.imread(image_r)
    # Validate image format
    N, M, D = np.int32(image.shape)
    assert D == 3, "Error: input must be 3-channel RGB image"
    print("Processing %d x %d RGB image" % (M, N))

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
        # if verbose:
        print("ITERATION %i" % iteration)

        ### Step 1: estimate (h, v) subproblem

        # subproblem 1 start time
        # s_time = time.time()

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

        # subproblem 1 end time
        # e_time = time.time()
        # step_1 = step_1 + e_time - s_time
        # if verbose:
        #     print("-subproblem 1: estimate (h,v)")
        #     print("--time: %f (s)" % (e_time - s_time))

        ### Step 2: estimate S subproblem

        # subproblem 2 start time
        # s_time = time.time()

        # compute dxhp + dyvp
        dxhp[:, 0:1, :] = h[:, M - 1:M, :] - h[:, 0:1, :]
        dxhp[:, 1:M, :] = -(np.diff(h, 1, 1))
        dyvp[0:1, :, :] = v[N - 1:N, :, :] - v[0:1, :, :]
        dyvp[1:N, :, :] = -(np.diff(v, 1, 0))
        normin = dxhp + dyvp

        # fft_s = time.time()
        FS[:, :, 0] = np.fft.fft2(normin[:, :, 0])
        FS[:, :, 1] = np.fft.fft2(normin[:, :, 1])
        FS[:, :, 2] = np.fft.fft2(normin[:, :, 2])
        # fft_e = time.time()
        # step_2_fft += fft_e - fft_s

        # solve for S + 1 in Fourier domain
        denorm = 1 + beta * MTF;
        FS[:, :, :] = (FI + beta * FS) / denorm

        # inverse FFT to compute S + 1
        S[:, :, 0] = np.float32((np.fft.ifft2(FS[:, :, 0])).real)
        S[:, :, 1] = np.float32((np.fft.ifft2(FS[:, :, 1])).real)
        S[:, :, 2] = np.float32((np.fft.ifft2(FS[:, :, 2])).real)
        # step_2_fft += fft_e - fft_s

        # subproblem 2 end time
        # e_time = time.time()
        # step_2 = step_2 + e_time - s_time
        # if verbose:
        #     print("-subproblem 2: estimate S + 1")
        #     print("--time: %f (s)" % (e_time - s_time))
        #     print("")

        # update beta for next iteration
        beta *= kappa
        iteration += 1

    # Rescale image
    S = (S - np.min(S)) / np.ptp(S)
    S = S * 255.0

    return S


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to image", type=str)
    args = parser.parse_args()
    image_path = args.path

    # img_path = join(dir_path, "images", "pano", "image{:04d}.png".format(readout_idx))
    if not exists(image_path):
        print(image_path, " could not be found")
        exit()

    # Load image
    # img = imread(image_path)

    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # img = cv2.blur(img, (10, 10))

    # https://github.com/kjzhang/kzhang-cs205-l0-smoothing
    s_image = l0_smoothing(image_path, 2.0, 0.01)
    print("1:", s_image.shape, type(s_image), s_image[0].dtype, np.min(s_image), np.max(s_image))

    cv2.imwrite(basename(image_path) + "_l0_smooth.jpg", s_image)

    # https://theailearner.com/tag/non-max-suppression/
    # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    # s_image = cv2.normalize(s_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print("2:", s_image.shape, type(s_image), s_image[0].dtype, np.min(s_image), np.max(s_image))
    edges = cv2.Canny(s_image, 50, 200, L2gradient=True)
    print("3:", edges.shape, type(edges), edges[0].dtype, np.min(edges), np.max(edges))
    cv2.imwrite(basename(image_path) + "_edges.jpg", edges)
