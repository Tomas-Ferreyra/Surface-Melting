#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:38:22 2025

@author: tomasferreyrahauchar
"""
import imageio.v2 as imageio
from tqdm import tqdm
from time import time

import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import remove_small_objects, binary_dilation, disk, skeletonize, binary_closing, remove_small_holes, binary_opening, binary_erosion
from skimage.segmentation import mark_boundaries, felzenszwalb
from skimage.filters import try_all_threshold, gaussian
#%%
t1 = time()

vid = imageio.get_reader('Documents/Dodecahedro/Calibration/HERO9 BLACK/GX010176.MP4', 'ffmpeg') # 5000 last frame

immed = []
# for i in tqdm(range(30,200)):
for i in tqdm(range(11500,12001)):
    fima = vid.get_data(i)
    immed.append( fima )

immed = np.median( immed, axis=0 )

t2 = time()
print(t2-t1)

np.shape( immed )
#%%
c = 0
n = 2950
fima = vid.get_data(n)

plt.figure()
plt.imshow( fima[:,:,c] )
plt.show()
# plt.figure()
# plt.imshow( immed[:,:,c] )
# plt.show()

plt.figure()
plt.imshow( fima[:,:,c] - immed[:,:,c] )
plt.colorbar()
plt.show()

# np.mean(fima[:,:,c] - immed[:,:,c]), np.mean(np.sqrt((fima[:,:,c] - immed[:,:,c])**2))

#%%

c = 2
fima = vid.get_data( 12000 )
plt.figure()
plt.imshow( fima )
plt.show()
# fima = vid.get_data( 12001 )
# plt.figure()
# plt.imshow( fima )
# plt.show()
# fima = vid.get_data( 12010 )
# plt.figure()
# plt.imshow( fima )
# plt.show()



# plt.figure()
# plt.imshow( fima[:,:,c] )
# plt.show()

# plt.figure()
# plt.imshow( fima[:,:,c] - immed[:,:,c] )
# plt.show()


#%%
t1 = time()

vid1 = imageio.get_reader('Documents/Dodecahedro/Calibration/HERO9 BLACK/GX010177.MP4', 'ffmpeg') # 16994 last frame, 24fps

immed_np = []
# for i in tqdm(range(30,200)):
for i in tqdm(range(16300, 16400)):
    fima = vid1.get_data(i)
    immed_np.append( fima )

immed_np = np.median( immed_np, axis=0 )

t2 = time()
print(t2-t1)

# np.shape( immed )
#%%

for i in range(16616,16636,1):
    plt.figure()
    plt.imshow( vid1.get_data( i ) )
    plt.title(i)
    plt.show()

#%%
plt.figure()
plt.imshow( immed_np[:,:,0] )
plt.show()
plt.figure()
plt.imshow( immed_np[:,:,1] )
plt.show()
plt.figure()
plt.imshow( immed_np[:,:,2] )
plt.show()
#%%

# plt.figure()
# plt.imshow( (vid1.get_data( 5280 ) - immed_np)[:,:,0]  )
# plt.colorbar()
# plt.show()
# plt.figure()
# plt.imshow( (vid1.get_data( 5280 ) - immed_np)[:,:,1]  )
# plt.colorbar()
# plt.show()
# plt.figure()
# plt.imshow( (vid1.get_data( 5280 ) - immed_np)[:,:,2]  )
# plt.colorbar()
# plt.show()

#%%
ice = (vid1.get_data( 5280 ) - immed_np)[:,:,0]
posi = binary_closing(ice>40, disk(2))

plt.figure()
plt.imshow( vid1.get_data( 5280 ) )
plt.show()
plt.figure()
plt.imshow( ice )
plt.show()

# plt.figure()
# plt.imshow( ice>40 )
# # plt.hist( ice.flatten(), bins=50 )
# plt.show()

# plt.figure()
# plt.imshow( mark_boundaries( vid1.get_data( 5280 ), posi) )
# # plt.hist( ice.flatten(), bins=50 )
# plt.show()

#%%
ice = (vid1.get_data( 6840 ) - immed_np)[:,:,0]
# ice = (vid1.get_data( 5280 ) - immed_np)[:,:,0]
# posi = binary_closing(ice>40, disk(2))

plt.figure()
plt.imshow( vid1.get_data( 6840 ) )
# plt.imshow( vid1.get_data( 5280 ) )
plt.show()

plt.figure()
plt.imshow( ice )
plt.show()

# scale = 5e5
# sigma = 0.5
# min_size = 10
# divs = felzenszwalb(ice, scale=scale, sigma=sigma, min_size=min_size)

# plt.figure()
# plt.imshow( divs )
# plt.colorbar()
# plt.show()

fig, ax = try_all_threshold(ice, figsize=(10, 8), verbose=False)
plt.show()


#%%

n = 16621
ref = vid1.get_data( n )

# plt.figure()
# plt.imshow( ref[:,:,0] )
# plt.colorbar()
# plt.show()
# plt.figure()
# plt.imshow( vid1.get_data( 5520 )[:,:,:] )
# plt.colorbar()
# plt.show()
# # plt.figure()
# # plt.imshow( vid1.get_data( 16608 )[:,:,:] )
# # # plt.colorbar()
# # plt.show()

# plt.figure()
# plt.imshow( (vid1.get_data( 5520 ) - ref * 1.)[:,:,0] )
# plt.colorbar()
# plt.show()

ise = (vid1.get_data( 5520 ) )[:,:,0]
plt.figure()
plt.imshow(ise)
plt.show()

fig, ax = try_all_threshold(ise, figsize=(10, 8), verbose=False)
plt.show()





#%%



#%%
t1 = time()

vid2 = imageio.get_reader('Documents/Dodecahedro/Calibration/HERO9 BLACK/GX010178.MP4', 'ffmpeg') # 13649 last frame, 24fps

immed_np = []
for i in tqdm(range(13470, 13520)):
    fima = vid2.get_data(i)
    immed_np.append( fima )

immed_np = np.median( immed_np, axis=0 )

immed_p = []
for i in tqdm(range(12400, 12450)):
    fima = vid2.get_data(i)
    immed_p.append( fima )

immed_p = np.median( immed_p, axis=0 )

t2 = time()
print(t2-t1)
#%%

# for i in range(13400,13550,10):
#     plt.figure()
#     # plt.imshow( vid2.get_data(i)[:,:,0] )
#     plt.imshow( vid2.get_data(i)[:,:,0] - immed_np[:,:,0] )
#     plt.colorbar()
#     plt.title(i)
#     plt.show()

i = 3600
ice = (vid2.get_data(i) - immed_np)[:,:,0]
plt.figure()
plt.imshow( ice )
plt.show()

fig, ax = try_all_threshold(ice, figsize=(10, 8), verbose=False)
plt.show()


#%%

# for i in range(6610,6640,3):
#     plt.figure()
#     plt.imshow( vid2.get_data(i)[:,:,:] )
#     # plt.imshow( vid2.get_data(i)[:,:,0] - immed_p[:,:,0] )
#     # plt.colorbar()
#     plt.title(i)
#     plt.show()

i = 6622
ice = (vid2.get_data(i) - immed_p)[:,:,0]
plt.figure()
plt.imshow( ice )
plt.colorbar()
plt.show()
plt.figure()
plt.imshow( vid2.get_data(i) )
plt.colorbar()
plt.show()

fig, ax = try_all_threshold(ice, figsize=(10, 8), verbose=False)
plt.show()

#%%
from scipy.ndimage import rotate
from scipy.signal import convolve, convolve2d, fftconvolve
from skimage.filters import gaussian
#%%
# =============================================================================
# Intento analisis refleccion
# =============================================================================
t1 = time()
# im = imageio.imread('Documents/Dodecahedro/Calibration/DSC_8676.jpeg')[800:3600,1700:7400,1]
im = imageio.imread('Documents/Dodecahedro/Calibration/DSC_8633.jpeg')[800:3600,1700:7400,1]
imr = rotate(im, 18.2)[2100:2840]
imr = gaussian(imr,0)
img = gaussian(imr,2)
t2 = time()
t2-t1
#%%

ny,nx = np.shape(img)
disp = []
for i in range(nx):
    linn = np.pad( imr[:,i], 0)
    ling = np.pad( img[:,i], 0)
    disp.append( np.argmax( np.convolve(ling, ling) ) - (len(ling)-1) )

disp = np.array(disp)
x = np.arange(nx)

# print(disp)

plt.figure()
plt.imshow( img )
plt.plot( x, ny/2+disp, 'r-' )
plt.show()


#%%

plt.figure()
plt.imshow( imr )
plt.show()
plt.figure()
plt.imshow( img )
plt.show()


plt.figure()

# plt.plot(linn, '-' )
# plt.plot(linn[::-1], '-' )

x = np.arange(len(ling))
plt.plot(x,ling, '-' )
plt.plot(x+disp,ling[::-1], '-' )

# plt.plot( np.convolve(linn, linn), '-' )
# plt.plot( np.arange( len(np.convolve(linn, linn))) - 859, np.convolve(ling, ling), '.-' )

plt.grid()
plt.show()

# fig, ax = try_all_threshold(ice, figsize=(10, 8), verbose=False)

#%%
t1 = time()
# cimg = fftconvolve(img, img, mode='same', axes=0)
cimg = fftconvolve(img, img, mode='same', axes=None)
t2 = time()
print(t2-t1)

plt.figure()
plt.imshow( img, cmap='gray' )
plt.plot( np.argmax(cimg, axis=0), '-' )
plt.show()

plt.figure()
plt.imshow( cimg, cmap='gray' )
plt.plot( np.argmax(cimg, axis=0), '-' )
plt.show()

#%%
# =============================================================================
# Experiments
# =============================================================================
# 1 Hz
vid = imageio.get_reader('Documents/Dodecahedro/Gopro experiments/GX010187.MP4', 'ffmpeg') 
# experiment starts 1280 until end (16994)
#%%
i = 1300
im = vid.get_data(i)
    
plt.figure()
plt.imshow( im )
plt.title(i)
plt.show()

#%%

# bac = vid.get_data(500)[:,:,2]
im1 = vid.get_data(1300)[:,:,2]
im2 = vid.get_data(1400)[:,:,2]

sigma = 0

plt.figure()
plt.imshow( ( gaussian(im1,sigma) - gaussian(im2,sigma) ) )
plt.show()

plt.figure()
plt.imshow( gaussian(im1,sigma) )
plt.show()

plt.figure()
plt.imshow( gaussian(im2,sigma) )
plt.show()



#%%






