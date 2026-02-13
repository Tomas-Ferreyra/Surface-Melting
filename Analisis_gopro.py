#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:38:22 2025

@author: tomasferreyrahauchar
"""
import imageio.v2 as imageio
import cv2
from tqdm import tqdm
from time import time

import numpy as np
import matplotlib.pyplot as plt

# from skimage.morphology import remove_small_objects, binary_dilation, disk, skeletonize, binary_closing, remove_small_holes, binary_opening, binary_erosion
from skimage.segmentation import mark_boundaries, felzenszwalb
from skimage.filters import gaussian, roberts, sobel # try_all_threshold

def normalize(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

#%%
# =============================================================================
# Experiments seen from below
# =============================================================================

def background_import(file_path, file, colorconv=cv2.COLOR_BGR2GRAY , n_mean=1000):
    bvid = cv2.VideoCapture(file_path+back_file+'.MP4') # 24fps, start 632
    blen = int(bvid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    bvid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    f1 = bvid.read()[1]
    ny,nx,_ = np.shape( f1 ) 
        
    bvid.set(cv2.CAP_PROP_POS_FRAMES, blen-n_mean)
    
    mean_frames = np.zeros( np.shape( cv2.cvtColor(f1, colorconv) )  )
    
    for i in tqdm(range(n_mean)):
        frame = bvid.read()[1]
        gray = cv2.cvtColor(frame, colorconv)
        
        mean_frames += gray    
    
    background = mean_frames / n_mean
    return background, ny,nx

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/'
back_file = 'GX020240'
ccc = cv2.COLOR_BGR2YCrCb
background, ny, nx = background_import(file_path, back_file, colorconv=ccc, n_mean=100)

#%%

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/'
file = 'GX010240'

# vid1 = imageio.get_reader(file_path+file+'.MP4', 'ffmpeg') 
vid = cv2.VideoCapture(file_path+file+'.MP4') # 24fps, start 632

vlen = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

print(vlen, ny,nx)



#%%


#%%

# i = 632
i = 2957

vid.set(cv2.CAP_PROP_POS_FRAMES, i)


N = 1
alf = np.zeros((ny,nx,N))

for i in range(N):
    fr = vid.read()[1]

    bb = np.round(background).astype('int')    

    frame_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    frame_lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
    frame_hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)

    # framb = np.abs(frame - background)
    # framb = np.abs( frame_lab - background )
    # framb = np.abs( frame_gray - background )
    
    fb = frame_gray-bb
    fb = (fb - np.min(fb)) / (np.max(fb) - np.min(fb)) *255
    fb = np.round(fb).astype('int')    
    
    # gy,gx = np.gradient( gaussian(framb, 2.5) )
    
    # alf[:,:,i] = framb
    
    # plt.figure()
    # plt.imshow( frame_gray, cmap='gray' )
    # plt.show()
    # plt.figure()
    # plt.imshow( bb, cmap='gray' )
    # plt.show()
    # plt.figure()
    # plt.imshow( fb , cmap='gray' )
    # plt.show()

    # plt.figure()
    # plt.imshow( roberts(fr) )
    # plt.show()
    # plt.figure()
    # plt.imshow( sobel(fr) )
    # plt.show()

    # plt.figure()
    # plt.imshow( frame_gray, cmap='gray' )
    # plt.show()  
    
    # plt.figure()
    # # plt.imshow( frame_lab[:,:,0], cmap='gray' )
    # # plt.imshow( background[:,:,0], cmap='gray' )
    # plt.imshow( framb[:,:,0], cmap='gray' )
    # plt.title('LAB-0')
    # plt.show()    
    # plt.figure()
    # # plt.imshow( frame_lab[:,:,1], cmap='gray' )
    # # plt.imshow( background[:,:,1], cmap='gray' )
    # plt.imshow( framb[:,:,1], cmap='gray' )
    # plt.title('LAB-1')
    # plt.show()    
    # plt.figure()
    # # plt.imshow( frame_lab[:,:,2], cmap='gray' )
    # # plt.imshow( background[:,:,2], cmap='gray' )
    # plt.imshow( framb[:,:,2], cmap='gray' )
    # plt.title('LAB-2')
    # plt.show()    

    plt.figure()
    plt.imshow( frame_hsv[:,:,1], cmap='gray' )
    plt.title('HSV')
    plt.show()    

    # plt.figure()
    # plt.imshow( background, cmap='gray' )
    # plt.show()
    
    # plt.figure()
    # plt.imshow( framb, cmap='gray' )
    # plt.show()
    
# plt.figure()
# plt.imshow( framb > 20, cmap='gray' )
# plt.show()

# both = np.abs(gy) + np.abs(gy)

# plt.figure()
# plt.imshow( both , cmap='gray') #, vmax=5 )
# plt.show()
# # plt.figure()
# # plt.imshow( np.abs(gx), cmap='gray', vmax=5 )
# # plt.show()


# plt.figure()
# plt.imshow( np.std(alf,axis=2) )
# plt.title(f'N = {N}')
# plt.show()

#%%

convert = cv2.COLOR_BGR2YCrCb

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/'
back_file = 'GX020240'
background, ny, nx = background_import(file_path, back_file, colorconv=convert, n_mean=100 )

#%%
bb = np.round(background).astype('int')    

frame_c = cv2.cvtColor(fr, convert)

c = 0


plt.figure()
plt.imshow( frame_c[:,:,c], cmap='gray' )
plt.show()
plt.figure()
plt.imshow( bb[:,:,c], cmap='gray' )
plt.show()


#%%

# c = 1
# # plt.figure()
# # plt.imshow( gaussian(fb[:,:,c],3) - gaussian(fb[:,:,c],25) )
# # # plt.imshow( fb[:,:,0] )
# # plt.show()
# # plt.figure()
# # plt.imshow( normalize( roberts( gaussian(fb[:,:,c],5) ) ), vmax=.4 )
# # plt.show()
# plt.figure()
# plt.imshow( normalize( sobel( gaussian(fb[:,:,c],1) ) ), vmax=.4 )
# plt.show()
# # plt.figure()
# # plt.imshow( fb[:,:,2] )
# # plt.show()

c = 0
plt.figure()
# plt.imshow( frame_hsv[:,:,c], cmap='gray' )
# plt.imshow( background[:,:,c], cmap='gray' )
plt.imshow( np.abs(frame_hsv-background)[:,:,c], cmap='gray' )
plt.title('HSV')
plt.show()    

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






