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

if bool(config.getboolean('Fourier','active')):
    # Prepare plot
    fig, ax = plt.subplots(2, 2, dpi=1000)
    ax[0, 0].set_title('Fourier spectrum')
    ax[0, 1].set_title('Fourier spectrum filtered')
    ax[1, 0].set_title('Image')
    ax[1, 1].set_title('Image filtered')

    #Fourier trafo
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
        if not gap/2 < _k < width/2:
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
    ax[1, 1].imshow(np.stack((orig_imgr, orig_imgg, orig_imgb), axis=-1).real.astype(int))
    plt.show()


if bool(config.getboolean('Red Line', 'active')):
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

    # save data
    save_path = path.parent / (path.stem + '_sceleton.csv')
    np.savetxt(save_path, filtered_img, fmt='%1i', delimiter=',')
    fig.savefig(path.parent / (path.stem + '_filter_plot'))
    # Todo: save config
