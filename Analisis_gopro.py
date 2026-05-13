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
i = 3000 #2957

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
    plt.imshow( frame_hsv[:,:,0], cmap='gray' )
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



#%%






#%%
















#%%
# =============================================================================
# New
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

from tqdm import tqdm
from time import time
from itertools import combinations

from skimage.filters import gaussian
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import remove_small_objects

def pick_points_on_figure(x, y, im, fig_num, max_picks=3):
    labels = ["(0,0)", "(1,0)", "(0,1)"]
    picked_points = []

    fig, ax = plt.subplots()
    ax.set_title(f"Figure {fig_num + 1}: Pick {max_picks} points")
    ax.imshow(im, cmap='gray')
    sc = ax.scatter(x, y, picker=True)

    def on_pick(event):
        if event.artist != sc:
            return
        ind = event.ind[0]
        picked_x = x[ind]
        picked_y = y[ind]

        # if (picked_x, picked_y) not in picked_points:
        idx = len(picked_points)
        picked_points.append(int(ind))
        ax.plot(picked_x, picked_y, 'ro')

        # Add label near the point
        ax.text(picked_x, picked_y, f" {labels[idx]}")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if len(picked_points) >= max_picks:
            fig.canvas.mpl_disconnect(cid)
            plt.pause(1)
            plt.close(fig)


    cid = fig.canvas.mpl_connect('pick_event', on_pick)

    while plt.fignum_exists(fig.number):
        plt.pause(0.1)  # pause briefly to process events

    return picked_points


def iscollinear(points):
    """
    Parameters
    ----------
    points : (N,M,2) array
        N sets of M points to check (in 2D).

    Returns
    -------
    array of booleans.
    """
    check = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        k = points[i]
        vecdiv = (k-k[0]) * (k[1]-k[0])[::-1]
        ks = np.diff(vecdiv, axis=1)[:, 0]

        check[i] = np.all(ks == 0)

    return check


def find_basis(out, position, gridpos, eps, lim_ind):
    order = np.argsort(np.sum((out - position)**2, axis=1))
    orout, orgridi = out[order], gridpos[order]

    for i in range(3, len(orout)+1):
        subs = list(combinations(np.arange(i), 3))
        subs_pts, subs_grid = orout[subs], orgridi[subs]
        is_col = iscollinear(subs_grid)
        subs_pts, subs_grid = subs_pts[~is_col], subs_grid[~is_col]

        if len(subs_pts) > 0:
            ibase = np.argmin(np.sum((subs_pts - position)**2, axis=(1, 2)))
            basep, baseg = subs_pts[ibase], subs_grid[ibase]
            break

    if len(subs_pts) > 0:
        vp1, vp2 = basep[1] - basep[0], basep[2] - basep[0]
        vg1, vg2 = baseg[1] - baseg[0], baseg[2] - baseg[0]
        vpoi = position - basep[0]

        a = np.linalg.det(np.vstack((vp1, vp2)))
        b = np.linalg.det(np.vstack((vp1, vpoi)))
        c = np.linalg.det(np.vstack((vpoi, vp2)))

        pti = c/a * vg1 + b/a * vg2 + baseg[0]
        newind = np.round(pti)
        err = np.sqrt(np.sum((newind - pti)**2))

        belongs_grid = (err < eps) and (newind[0] >= lim_ind[0]) and (
            newind[0] <= lim_ind[1]) and (newind[1] >= lim_ind[2]) and (newind[1] <= lim_ind[3])
    else:
        newind = np.zeros(2)*np.nan
        belongs_grid = False

    return newind, belongs_grid

def find_indices(points, known, eps=0.25, lim_ind=[-1e5, 1e5, -1e5, 1e5], gps=[1,1]):
    """
    Parameters
    ----------
    points : (N,2)-array
        Points to match in  a grid.
    known : list of 3 integers 
        Indeces in points for (0,0), (1,0) and (0,1).
    eps : TYPE, optional
        DESCRIPTION. The default is 0.25.
    lim_ind : list length 4, optional
        min and max indices for [left,right,bottom,top]
    gps : [±1,±1]
        Direction the first and/or second point (for starting grid) is going
    Returns
    -------
    (M,2)-array
        position of points in grid.
    (M,2)-array
        grid position of points.
    """
    N, col = np.shape(points)
    if col != 2:
        return 'Points need to be 2D'

    pts = np.copy(points)
    out, gridpos = np.zeros((3, 2)), np.zeros((3, 2))
    out[0], out[1], out[2] = pts[known[0]], pts[known[1]], pts[known[2]]
    gridpos[1, 0], gridpos[2, 1] = gps[0], gps[1]

    pts = np.delete(pts, known, axis=0)

    itere = 0
    while len(pts) > 0 and itere <= N:
        itere += 1
        med = np.mean(out, axis=0)
        iclosest = np.argmin(np.sum((pts - med)**2, axis=1))
        position = pts[iclosest]
        pts = np.delete(pts, iclosest, axis=0)

        newind, belongs_grid = find_basis(out, position, gridpos, eps, lim_ind)
        if belongs_grid:
            out = np.vstack((out, position))
            gridpos = np.vstack((gridpos, newind))

    return out, gridpos.astype('int64')

#%%
save_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Gopro recordings/'
folder = '25 mm/Rep0-5kg-1hz/'

files = glob.glob( save_path + folder + 'GX*' )
vids = [ cv2.VideoCapture( files[i] ) for i in range(len(files)) ]
vlens = [int(vids[i].get(cv2.CAP_PROP_FRAME_COUNT)) for i in range(len(vids))]
#%%

l = 1
bframes = 100 

backs = np.zeros( (2160, 3840, bframes) )

vids[l].set(cv2.CAP_PROP_POS_FRAMES, vlens[-1] - bframes )
for i in tqdm(range( bframes )):      
    im = np.array( vids[l].read()[1] )[:,:,::-1]
    backs[:,:,i] = im[:,:,1]
    
    
backs = np.median(backs, axis=2)


#%%
l = 0
fr = 2000

vids[l].set(cv2.CAP_PROP_POS_FRAMES, fr)  
im = np.array( vids[l].read()[1] )[:,:,::-1]
imb = im[:,:,1] - backs 

plt.figure()
plt.imshow( imb )
plt.colorbar()
plt.show()
# plt.figure()
# plt.imshow( backs )
# plt.show()
# plt.figure()
# plt.imshow( im[:,:,1] )
# plt.show()


#%%

plt.figure()
plt.imshow( imb > 30 )
plt.show()

#%%
# =============================================================================
# Calibration
# =============================================================================

save_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Gopro recordings/'
folder = '25 mm/'

cal = 'Calibration_4.MP4'

vcal = cv2.VideoCapture( save_path + folder + cal )
lvc = int(vcal.get(cv2.CAP_PROP_FRAME_COUNT))


#%%

fr = 682 #0,319,590,682
thres_1 = 0.52
thres_2 = -0.05
min_size = 5

t1 = time()

vcal.set(cv2.CAP_PROP_POS_FRAMES, fr)  
im = np.array( vcal.read()[1] )[:,:,::-1]
im = im[:,:,1] / 255.

img = gaussian(im, 1)
im2 = gaussian(im, 10)

labe = label(im2 > thres_1)
prop = regionprops_table(labe, properties=['area'])['area']
reg = np.argmax(prop)+1
fil = labe == reg

dots = remove_small_objects( ((img-im2) * fil) < thres_2 , min_size=min_size)

lab = label(dots)
props = regionprops(lab )
cy = np.array([(props[i].centroid)[0] for i in range(len(props))])
cx = np.array([(props[i].centroid)[1] for i in range(len(props))])

# cy = np.array([(props[i].centroid_weighted)[0] for i in range(len(props))])
# cx = np.array([(props[i].centroid_weighted)[1] for i in range(len(props))])

t2 = time()

t2-t1

#%%

plt.figure()
plt.imshow( dots )
plt.colorbar()
plt.show()
# plt.figure()
# plt.imshow( fil )
# plt.colorbar()
# plt.show()
plt.figure()
plt.imshow( img  )
# plt.plot( cx,cy, 'r.' )
plt.colorbar()
plt.show()

#%%

plt.figure()
plt.imshow( im2 > 0.52 )
# plt.imshow( im2 - img )
plt.colorbar()
plt.show()

#%%

points = pick_points_on_figure(cx, cy, img, 0, max_picks=3)

#%%
points = [3111, 3038, 3162] # [3026, 2973, 2974] # [3023, 3022, 2972]
pp = np.vstack((cx,cy)).T
out, gridpos = find_indices( pp, points, eps=0.2, lim_ind=[-27, 27, -54, 54], gps=[-1,1] )


plt.figure()
plt.gca().invert_yaxis()
# plt.imshow( img )

plt.plot(pp[:, 0], pp[:, 1], 'r.', markersize=10)
plt.plot(out[:, 0], out[:, 1], 'b.', markersize=8)
# plt.plot( pp[known,0], pp[known,1], '.' )
for j in range(len(out)):
    texto = "({:.0f},{:.0f})".format(gridpos[j, 0], gridpos[j, 1])
    plt.text(out[j, 0], out[j, 1], texto, fontsize=9)
plt.title(i)
plt.grid()
plt.show()



#%%
# =============================================================================
# IR camera
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import cv2

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/IR camera/IR-'
file = '10kg-4HZ'

vid = cv2.VideoCapture(file_path+file+'.MP4') # 24fps, start 632
vlen = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

vlen
#%%

i = 0
vid.set(cv2.CAP_PROP_POS_FRAMES, i)

im = vid.read()[1]

plt.figure()
plt.imshow( im )
plt.show()

#%%

plt.figure()
for h in range(3):
    plt.plot( im[6:474,655,h], label=h )
plt.legend()
plt.show()




#%%











