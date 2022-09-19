import cv2
import scipy.fft
from skimage.filters import difference_of_gaussians, window
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np

img = cv2.cvtColor(cv2.imread('./Data/SB_20220228_007_001.tiff'), cv2.COLOR_BGR2RGB)

plt.figure(dpi=1000)

fft_imgr = np.fft.fft2(img[:, :, 0])
fft_imgr = scipy.fft.fftshift(fft_imgr)
fft_imgg = np.fft.fft2(img[:, :, 1])
fft_imgg = scipy.fft.fftshift(fft_imgg)
fft_imgb = np.fft.fft2(img[:, :, 2])
fft_imgb = scipy.fft.fftshift(fft_imgb)

plt.imshow(np.log(np.absolute(fft_imgb)), cmap='magma', aspect='auto')
plt.show()

angle = 0.1
for k in range(img.shape[1]):
    _k = abs(k - img.shape[1] / 2)
    if not 5 < _k < 200:
        pass
    else:
        half = int(img.shape[0] / 2)
        value = int(angle * _k)
        for j in range(half-value, half + value):
            fft_imgr[j, k] = 0
            fft_imgg[j, k] = 0
            fft_imgb[j, k] = 0


plt.imshow(np.log(np.absolute(fft_imgb)), cmap='magma', aspect='auto')
plt.show()


fig, ax = plt.subplots(1, 2, dpi=1000)
ax[0].imshow(img)

orig_imgr = np.fft.ifft2(scipy.fft.fftshift(fft_imgr))
orig_imgg = np.fft.ifft2(scipy.fft.fftshift(fft_imgg))
orig_imgb = np.fft.ifft2(scipy.fft.fftshift(fft_imgb))
ax[1].imshow(np.stack((orig_imgr, orig_imgg, orig_imgb), axis=-1).real.astype(int))
plt.show()
exit()

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].imshow(img)
lowpass = ndi.gaussian_filter1d(img, 3, axis=1)
lowpass2 = ndi.gaussian_filter1d(img, 10, axis=1)
ax[1].imshow(img - (lowpass - lowpass2))
plt.show()

fft_imgr = np.fft.fft2((img - (lowpass - lowpass2))[:, :, 0])
fft_imgr = scipy.fft.fftshift(fft_imgr)
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].imshow(np.log(np.absolute(fft_imgr)), cmap='magma', aspect='auto')

fft_imgr = np.fft.fft2(img[:,:,0])# * window('hann', img.shape)) #[:, int(fft_img.shape[1]*2/5):int(fft_img.shape[1]*3/5)])
fft_imgr = scipy.fft.fftshift(fft_imgr)
ax[1].imshow(np.log(np.absolute(fft_imgr)), cmap='magma', aspect='auto')
plt.show()
exit()

fft_imgr = np.fft.fft2(img[:,:,0])# * window('hann', img.shape)) #[:, int(fft_img.shape[1]*2/5):int(fft_img.shape[1]*3/5)])
fft_imgr = scipy.fft.fftshift(fft_imgr)
plt.imshow(np.log(np.absolute(fft_imgr)), cmap='magma', aspect='auto')
plt.show()
fft_imgg = np.fft.fft2(img[:,:,1])# * window('hann', img.shape))
fft_imgg = scipy.fft.fftshift(fft_imgg)
plt.imshow(np.log(np.absolute(fft_imgg)), cmap='magma', aspect='auto')
plt.show()
fft_imgb = np.fft.fft2(img[:,:,2])# * window('hann', img.shape))
fft_imgb = scipy.fft.fftshift(fft_imgb)
plt.imshow(np.log(np.absolute(fft_imgb)), cmap='magma', aspect='auto')
plt.show()


orig_imgr = np.fft.ifft2(scipy.fft.fftshift(fft_imgr))
orig_imgg = np.fft.ifft2(scipy.fft.fftshift(fft_imgr))
orig_imgb = np.fft.ifft2(scipy.fft.fftshift(fft_imgr))
plt.imshow(np.concatenate((orig_imgr, orig_imgg, orig_imgb)).real)
plt.show()