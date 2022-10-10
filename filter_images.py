import cv2
import scipy.fft
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
import configparser
from pathlib import Path

config = configparser.ConfigParser()
config.read('filter_config.ini')

path = Path(config['Data']['dir']) / config['Data']['origin_file']
img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

if bool(config.getboolean('Fourier', 'active')):
    # Prepare plot
    fig, ax = plt.subplots(2, 2, dpi=1000)
    ax[0, 0].set_title('Fourier spectrum')
    ax[0, 1].set_title('Fourier spectrum filtered')
    ax[1, 0].set_title('Image')
    ax[1, 1].set_title('Image filtered')

    # Fourier trafo
    fft_imgr = np.fft.fft2(img[:, :, 0])
    fft_imgr = scipy.fft.fftshift(fft_imgr)
    fft_imgg = np.fft.fft2(img[:, :, 1])
    fft_imgg = scipy.fft.fftshift(fft_imgg)
    fft_imgb = np.fft.fft2(img[:, :, 2])
    fft_imgb = scipy.fft.fftshift(fft_imgb)
    ax[0, 0].imshow(np.log(np.absolute(fft_imgb)), cmap='magma', aspect='auto')

    # Filter
    angle = float(config['Fourier']['slope'])
    width = int(config['Fourier']['width'])
    gap = int(config['Fourier']['gap'])
    for k in range(img.shape[1]):
        _k = abs(k - img.shape[1] / 2)
        if not gap/2 < _k < (img.shape[1]/2)*width:
            pass
        else:
            half = int(img.shape[0] / 2)
            value = int(angle * _k)
            for j in range(half-value, half + value):
                fft_imgr[j, k] = 0
                fft_imgg[j, k] = 0
                fft_imgb[j, k] = 0
    ax[0, 1].imshow(np.log(np.absolute(fft_imgb)), cmap='magma', aspect='auto')
    ax[1, 0].imshow(img)

    # Fourier Transform back to image space
    orig_imgr = np.fft.ifft2(scipy.fft.fftshift(fft_imgr))
    orig_imgg = np.fft.ifft2(scipy.fft.fftshift(fft_imgg))
    orig_imgb = np.fft.ifft2(scipy.fft.fftshift(fft_imgb))
    orig_img = np.stack((orig_imgr, orig_imgg, orig_imgb), axis=-1).real.astype(np.uint8)
    ax[1, 1].imshow(orig_img)
    plt.show()

    image2save = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path.parent / (path.stem + '_filtered.tiff')), image2save)


if bool(config.getboolean('Red Line', 'active')):
    if config.getboolean('Red Line', 'use_white_light'):
        path = path.parent / (path.stem + '_WL' + path.suffix)
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    # prepare plots
    fig, ax = plt.subplots(ncols=3, dpi=1000)
    ax[0].set_title('Original image')
    ax[1].set_title('Distance image')
    ax[2].set_title('Red filtered image')

    y, x = config.getint('Red Line', 'y'), config.getint('Red Line', 'x')
    ax[0].imshow(img)
    ax[0].plot(x, y, 'r+')

    red = img[y, x, :]
    threshold = config.getfloat('Red Line', 'threshold')
    max_brightness = config.getint('Red Line', 'max_brightness')
    shape = img.shape
    distance_img = np.empty((shape[0], shape[1]))
    filtered_img = np.empty((shape[0], shape[1]))
    for r in range(shape[0]):
        for c in range(shape[1]):
            color_distance = np.linalg.norm(red - img[r, c, :])
            distance_img[r, c] = color_distance
            brightness = np.mean(img[r, c, :])
            right_color = color_distance < threshold \
                and brightness < max_brightness
            filtered_img[r, c] = 1 if right_color else 0
    filtered_img_grey = np.stack((filtered_img, filtered_img, filtered_img), axis=2)

    ax[1].imshow(distance_img)
    ax[2].imshow(filtered_img_grey)
    plt.show()
    if config.getboolean('Red Line', 'smooth'):
        distance_img_blurred = ndi.gaussian_filter(distance_img, 4)
        filtered_img_grey_blurred = np.where(distance_img_blurred > threshold, 1.0, 0.0)
        fig, ax = plt.subplots(ncols=3, dpi=1000)
        ax[1].imshow(distance_img_blurred)
        ax[2].imshow(np.stack((filtered_img_grey_blurred, filtered_img_grey_blurred, filtered_img_grey_blurred), axis=2))
        plt.show()

    # save data
    np.savetxt(path.parent / (path.stem + '_sceleton.csv'), filtered_img, fmt='%1i', delimiter=',')
    # fig.savefig(path.parent / (path.stem + '_filter_plot'))

# save config used
with open(path.parent / (path.stem + '_filter_config.ini'), 'w') as f:
    config.write(f)

