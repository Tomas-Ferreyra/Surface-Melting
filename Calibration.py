#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:48:36 2025

@author: tomasferreyrahauchar
"""

import imageio.v2 as imageio
# import imageio.v3 as iio
from tqdm import tqdm
from time import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares, differential_evolution, curve_fit
from scipy.spatial import Delaunay
from scipy.stats import linregress
from scipy.ndimage import rotate

from skimage.morphology import remove_small_objects, binary_dilation, disk, skeletonize, binary_closing, remove_small_holes, binary_opening, binary_erosion
from skimage.measure import label, regionprops
from skimage.filters import try_all_threshold, threshold_otsu, gaussian, threshold_yen, threshold_mean
import skimage.filters.rank as sfr
# from skimage.segmentation import felzenszwalb, mark_boundaries

import discorpy.losa.loadersaver as losa
import discorpy.prep.preprocessing as prep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post
#%%

def order_points( centsx, centsy, max_iter=100, distance=20):
    puntos = np.hstack((np.array([centsx]).T, np.array([centsy]).T, np.array([[0]*len(centsy)]).T ))
    mima = [np.argmin(puntos[:,0]+puntos[:,1]), np.argmax(puntos[:,0]-puntos[:,1])]
    or_puntos = np.empty((0,3), int)

    itere = 0
    while len(puntos) > 0 and itere < max_iter:
        mima = [np.argmin(puntos[:,0]+puntos[:,1]), np.argmax(puntos[:,0]-puntos[:,1])]
        dists = []
        p1, p2 = puntos[mima][0], puntos[mima][1] 
        for i in range(len(puntos)):
            p3 = puntos[i]
            dist = np.linalg.norm( np.cross(p2-p1, p1-p3) ) / np.linalg.norm(p2-p1)
            dists.append(dist)
        dists = np.array(dists)
        fil = dists < distance
        orde = np.argsort( puntos[fil][:,0] )
        
        or_puntos = np.vstack((or_puntos, puntos[fil][orde])) 
        
        puntos = puntos[~fil]
        itere += 1

    if itere >= max_iter: print('Maximum iterations reached')
    return or_puntos

def dist(cgrx,cgry,point):
    px,py = point[0],point[1]
    return (cgrx - px)**2 + (cgry - py)**2 

def diference_grids(cgrx1,cgry1,cgrx2,cgry2, spx1,spy1,spx2,spy2):
    dists = []
    for i in range(-29,31):
        for j in range(-27,28):
            point = [j,i]
            parg1 = np.where( dist(cgrx1,cgry1,point) == 0)[0]
            parg2 = np.where( dist(cgrx2,cgry2,point) == 0)[0]
            
            dx = (spx1[parg1] ) - (spx2[parg2] )  
            dy = (spy1[parg1] ) - (spy2[parg2] )
            dists.append( (dx**2 + dy**2)[0] )
            
    dists = np.array(dists)
    return np.nansum(dists)


def abc(pxs,pys):
    mat = np.array([[pxs[0]**0,pxs[0]**1,pxs[0]**2],
                    [pxs[1]**0,pxs[1]**1,pxs[1]**2],
                    [pxs[2]**0,pxs[2]**1,pxs[2]**2]] )
    yp = np.array([pys]).T

    return np.matmul(np.linalg.inv(mat), yp)

def rotate_point(theta, cex, cey):
    rcx = np.cos(theta) * cex - np.sin(theta) * cey
    rcy = np.sin(theta) * cex + np.cos(theta) * cey
    return rcx, rcy
    
def dists2_cuad_point(a,b,c,x, point ):
    return (x - point[0])**2 + (a+b*x+c*x**2 - point[1])**2

def xroot(a,b,c,point):
    px,py = point[0], point[1]
    c1 = 2*a*b - 2*px - 2*b*py
    c2 = 2 + 2*b**2 + 4*a*c - 4*c*py
    c3 = 6*b*c 
    c4 = 4*c**2

    coff = [c4,c3,c2,c1] #[c1,c2,c3,c4]    

    root = np.roots(coff)
    indi = np.argmin( dists2_cuad_point(a, b, c, root, point) )
    xro = np.real(root[indi])
    return xro

def point_to_caud(a,b,c, point):
    xro = xroot(a, b, c, point)
    return dists2_cuad_point(a,b,c,xro, point )

def pol2(x,a,b,c):
    return a + b*x + c * x**2    

def pol3(x,a,b,c,d):
    return a + b*x + c*x**2 + d*x**3

def distspol_cuad_point(x,y, point ):
    return (x - point[0])**2 + (y - point[1])**2

#%%
dboth = 0.04233 #mm/px
t1 = time()

# # grid 1
gridf1 = imageio.imread('Documents/Dodecahedro/Calibration/Scan_1.jpg')
grid = imageio.imread('Documents/Dodecahedro/Calibration/Scan_1.jpg')[400:,100:6580]
gridb = grid > 240
gridn = remove_small_objects( np.pad(gridb,25), min_size=250)
gridn[9820:] = 0.0
gridj = binary_closing(gridn, disk(20))[25:-25,25:-25]

gridl = label(gridj)
props = regionprops(gridl)

centsy = np.array( [(props[i].centroid)[0] for i in range(len(props)) ] ) + 400
centsx = np.array( [(props[i].centroid)[1] for i in range(len(props)) ] ) + 100

sor_points = order_points(centsx, centsy, max_iter=200, distance=40) # en px
spx, spy = sor_points[:,0], sor_points[:,1]

dcx, dcy = spx[1:]-spx[:-1], spy[1:]-spy[:-1] 
din1 = np.argmax(dcx)
cxm, cym = int(spx[din1]) - 100, int(spy[din1]) - 400
imgg = gridj[ cym-150:cym+150, cxm-50:cxm+300 ]
cy,cx = regionprops(imgg*1)[0].centroid
cx1, cy1 = cx + cxm + 50, cy + cym + 251
# cx1, cy1 = cx + cxm + 0, cy + cym + 251

spx1, spy1 = np.insert(spx, din1+1, np.nan), np.insert(spy, din1+1, np.nan)
xgr, ygr = np.arange(55), np.arange(82)
xgr,ygr = np.meshgrid(xgr,ygr)
cgrx, cgry = xgr.flatten(), ygr.flatten()
cgrx1, cgry1 = cgrx - cgrx[din1+1], cgry - cgry[din1+1]

# # grid 2
gridf2 = imageio.imread('Documents/Dodecahedro/Calibration/Scan_2.jpg')[::-1,::-1]
# grid = imageio.imread('Documents/Dodecahedro/Calibration/Scan_2.jpg')[80:10100,100:6600]
grid = imageio.imread('Documents/Dodecahedro/Calibration/Scan_2.jpg')[10100:80:-1,6600:100:-1]
gridb = grid > 240
gridn = remove_small_objects( np.pad(gridb,25), min_size=350)
# # gridn[9820:] = 0.0
gridj = binary_closing(gridn, disk(20))[25:-25,25:-25]

gridl = label(gridj)
props = regionprops(gridl)

centsy = np.array( [(props[i].centroid)[0] for i in range(len(props)) ] ) + 107
centsx = np.array( [(props[i].centroid)[1] for i in range(len(props)) ] ) + 407

sor_points = order_points(centsx, centsy, max_iter=220, distance=40) # en px
spx, spy = sor_points[:,0], sor_points[:,1]

dcx, dcy = spx[1:]-spx[:-1], spy[1:]-spy[:-1] 
din2 = np.argmax(dcx)
cxm, cym = int(spx[din2]) - 407, int(spy[din2]) - 107
imgg = gridj[ cym-150:cym+150, cxm-50:cxm+300 ]
cy,cx = regionprops(imgg*1)[0].centroid
# cx2, cy2 = cx + cxm + 50, cy + cym - 69
cx2, cy2 = cx + cxm + 358, cy + cym - 42

spx2, spy2 = np.insert(spx, din2+1, np.nan), np.insert(spy, din2+1, np.nan)
xgr, ygr = np.arange(55), np.arange(84)
xgr,ygr = np.meshgrid(xgr,ygr)
cgrx, cgry = xgr.flatten(), ygr.flatten()
cgrx2, cgry2 = cgrx - cgrx[din2+1], cgry - cgry[din2+1]

t2 = time()
t2-t1
#%%

fin = None

plt.figure()
plt.imshow(gridf1, cmap='gray')
# plt.plot(spx1[:fin], spy1[:fin], 'r.')
plt.plot(cx1, cy1, 'g.')
plt.show()

plt.figure()
plt.imshow(gridf2, cmap='gray')
# plt.plot(spx2[:fin], spy2[:fin], 'r.')
plt.plot(cx2, cy2, 'g.')
plt.show()


# plt.figure()
# plt.plot(spx1[:fin] - cx1, spy1[:fin] - cy1,'r.')
# plt.plot(cx1-cx1, cy1-cy1,'b.')
# plt.grid()
# plt.gca().invert_yaxis()
# plt.show()
# plt.figure()
# plt.plot(spx2[:fin] - cx2, spy2[:fin] - cy2,'r.')
# plt.plot(cx2-cx2, cy2-cy2,'b.')
# plt.grid()
# plt.gca().invert_yaxis()
# plt.show()

plt.figure()
plt.plot(cgrx1[:fin], cgry1[:fin], '.')
# plt.plot(cgrx1[din1+1], cgry1[din1+1], '.')
plt.axis('equal')
# plt.grid()
# plt.gca().invert_yaxis()
# plt.show()
# plt.figure()
plt.plot(cgrx2[:fin], cgry2[:fin], '.')
plt.plot(cgrx2[din2+1], cgry2[din2+1], '.')
plt.axis('equal')
plt.grid()
plt.gca().invert_yaxis()
plt.show()

xgr1,ygr1 = (spx1[:fin] - cx1) * dboth, (spy1[:fin] - cy1) * dboth
xgr2,ygr2 = (spx2[:fin] - cx2) * dboth, (spy2[:fin] - cy2) * dboth

plt.figure()
plt.plot( xgr1, ygr1 ,'b.')
plt.plot( [0], [0],'y.')

plt.plot( xgr2, ygr2,'r.')
plt.plot( [0], [0],'g.')

# plt.axis('equal')
plt.grid()
plt.gca().invert_yaxis()
plt.show()

#%%


def lessq(val):
    rgrx2 = np.cos(val[0]) * (spx2 - cx2) - np.sin(val[0]) * (spy2 - cy2)
    rgry2 = np.sin(val[0]) * (spx2 - cx2) + np.cos(val[0]) * (spy2 - cy2)
    
    return diference_grids(cgrx1,cgry1,cgrx2,cgry2, spx1-cx1,spy1-cy1, rgrx2*val[1] , rgry2*val[1] ) 
# (cgrx1, cgry1, rgrx2 * val[1], rgry2 * val[1])
    
t1 = time()

resid = least_squares(lessq, [-1.0,1.2])

t2 = time()
print(t2-t1)

resid, resid.x
#%%
def diference_grids(cgrx1,cgry1,cgrx2,cgry2, spx1,spy1,spx2,spy2):
    dists = []
    for i in range(-29,31):
        for j in range(-27,28):
            point = [j,i]
            parg1 = np.where( dist(cgrx1,cgry1,point) == 0)[0]
            parg2 = np.where( dist(cgrx2,cgry2,point) == 0)[0]
            
            dx = (spx1[parg1] ) - (spx2[parg2] )  
            dy = (spy1[parg1] ) - (spy2[parg2] )
            dists.append( (dx**2 + dy**2)[0] )
            
    dists = np.array(dists)
    return np.nansum(dists)

def lessq(val):
    rgrx2_ = np.cos(val[0]) * xgr2 - np.sin(val[0]) * ygr2
    rgry2_ = np.sin(val[0]) * xgr2 + np.cos(val[0]) * ygr2
    
    topx = val[1] * rgrx2_ + val[2] * rgry2_ + val[3]
    topy = val[4] * rgrx2_ + val[5] * rgry2_ + val[6]
    bot =  val[7] * rgrx2_ + val[8] * rgry2_ + 1
    
    rgrx2 = topx / bot
    rgry2 = topy / bot
    
    return diference_grids(cgrx1,cgry1,cgrx2,cgry2,  xgr1,ygr1, rgrx2, rgry2 ) 
# (cgrx1, cgry1, rgrx2 * val[1], rgry2 * val[1])
    
t1 = time()

# resid = least_squares(lessq, [-0,1,0,0,0,1,0,0,0])

t2 = time()
print(t2-t1)
print(resid.x)

val = resid.x
rgrx2_ = np.cos(val[0]) * xgr2 - np.sin(val[0]) * ygr2
rgry2_ = np.sin(val[0]) * xgr2 + np.cos(val[0]) * ygr2

topx = val[1] * rgrx2_ + val[2] * rgry2_ + val[3]
topy = val[4] * rgrx2_ + val[5] * rgry2_ + val[6]
bot =  val[7] * rgrx2_ + val[8] * rgry2_ + 1

rgrx2 = topx / bot
rgry2 = topy / bot
    
plt.figure()
plt.plot( xgr1, ygr1 ,'b.')

plt.plot( xgr2, ygr2,'r.')
# plt.plot( [0], [0],'g.')

plt.plot( rgrx2, rgry2,'g.')

plt.plot( [0], [0],'y.')

# plt.axis('equal')
plt.grid()
plt.gca().invert_yaxis()
plt.show()

#%%
# =============================================================================
# New scans
# =============================================================================
db0th = 0.03522 #cm/px
t1 = time()

# # grid 1
gridf1 = imageio.imread('Documents/Dodecahedro/Calibration/Newscan_1.jpg')[:,:,2]
grid = imageio.imread('Documents/Dodecahedro/Calibration/Newscan_1.jpg')[28:1218,18:794,2]
gridb = grid > 200
# gridn = remove_small_objects( np.pad(gridb,25), min_size=250)
# gridn[9820:] = 0.0
gridj = binary_closing( np.pad(gridb,25), disk(3))[25:-25,25:-25]

gridl = label(gridj)
props = regionprops(gridl)

centsy = np.array( [(props[i].centroid)[0] for i in range(len(props)) ] ) + 28
centsx = np.array( [(props[i].centroid)[1] for i in range(len(props)) ] ) + 18

sor_points = order_points(centsx, centsy, max_iter=200, distance=10) # en px
spx, spy = sor_points[:,0], sor_points[:,1]
dcx, dcy = spx[1:]-spx[:-1], spy[1:]-spy[:-1] 
din1 = np.argmax(dcx)

def cent_dist(val):
    di0 = (val[0] - spx)**2  + (val[1] - spy)**2
    idi0 = np.argsort( di0 )
    return np.sum(di0[idi0[:8]] ) / di0[idi0[0]]

# bounds = ( (int(np.mean(spx)-300), int(np.mean(spx)+300)), (int(np.mean(spy)-300), int(np.mean(spy)+300)) )
bounds = ( (int(spx[din1])-50, int(spx[din1])+50), (int(spy[din1])-50, int(spy[din1])+50) )
redi = differential_evolution(cent_dist, bounds, integrality = [True,True])

x0 = redi.x
red = least_squares(cent_dist, x0, bounds=((x0[0]-10,x0[1]-10),(x0[0]+10,x0[1]+10)) )
cx1, cy1 = red.x[0], red.x[1]

spx1, spy1 = np.insert(spx, din1+1, np.nan), np.insert(spy, din1+1, np.nan)
xgr, ygr = np.arange(55), np.arange(82)
xgr,ygr = np.meshgrid(xgr,ygr)
cgrx, cgry = xgr.flatten(), ygr.flatten()
cgrx1, cgry1 = cgrx - cgrx[din1+1], cgry - cgry[din1+1]


t2 = time()
print(t2-t1)
t1 = time()

gridf2 = imageio.imread('Documents/Dodecahedro/Calibration/Newscan_2.jpg')[::-1,::-1,2]
grid = imageio.imread('Documents/Dodecahedro/Calibration/Newscan_2.jpg')[:8:-1,794:18:-1,2]
gridb = grid > 200
gridn = remove_small_objects( np.pad( gridb,25), min_size=4, connectivity=2)
gridn[:,:27] = 0.0
gridj = remove_small_objects( binary_closing( gridn, disk(3))[25:-25,25:-25], min_size=2 )

gridl = label(gridj)
props = regionprops(gridl)

centsy = np.array( [(props[i].centroid)[0] for i in range(len(props)) ] ) 
centsx = np.array( [(props[i].centroid)[1] for i in range(len(props)) ] ) + 47

sor_points = order_points(centsx, centsy, max_iter=200, distance=10) # en px
spx, spy = sor_points[:,0], sor_points[:,1]
dcx, dcy = spx[1:]-spx[:-1], spy[1:]-spy[:-1] 
din2 = np.argmax(dcx)

def cent_dist(val):
    di0 = (val[0] - spx)**2  + (val[1] - spy)**2
    idi0 = np.argsort( di0 )
    return np.sum(di0[idi0[:8]] ) / di0[idi0[0]]

# bounds = ( (int(np.mean(spx)-300), int(np.mean(spx)+300)), (int(np.mean(spy)-300), int(np.mean(spy)+300)) )
bounds = ( (int(spx[din2])-50, int(spx[din2])+50), (int(spy[din2])-50, int(spy[din2])+50) )
redi = differential_evolution(cent_dist, bounds, integrality = [True,True])

x0 = redi.x
red = least_squares(cent_dist, x0, bounds=((x0[0]-10,x0[1]-10),(x0[0]+10,x0[1]+10)) )
cx2, cy2 = red.x[0], red.x[1]

spx2, spy2 = np.insert(spx, din2+1, np.nan), np.insert(spy, din2+1, np.nan)
xgr, ygr = np.arange(55), np.arange(82)
xgr,ygr = np.meshgrid(xgr,ygr)
cgrx, cgry = xgr.flatten(), ygr.flatten()
cgrx2, cgry2 = cgrx - cgrx[din2+1], cgry - cgry[din2+1]

fin = None
xgr1,ygr1 = (spx1[:fin] - cx1) * dboth, (spy1[:fin] - cy1) * dboth
xgr2,ygr2 = (spx2[:fin] - cx2) * dboth, (spy2[:fin] - cy2) * dboth

t2 = time()
print(t2-t1)
#%%

fin = 56

plt.figure()
plt.imshow(  gridf1, cmap='gray' )
# plt.imshow(  gridf2, cmap='gray' )
plt.colorbar()

plt.plot(centsx, centsy, '.')
# plt.plot(centsx[:fin], centsy[:fin], '.')
plt.plot(spx[:fin], spy[:fin], '.')
plt.plot( cx1, cy1, '.' )

plt.show()


plt.figure()

plt.plot( cgrx1, cgry1, 'o' )
plt.plot( cgrx2, cgry2, '.' )

plt.grid()
plt.axis('equal')
plt.gca().invert_yaxis()
plt.show()

plt.figure()
plt.plot( xgr1, ygr1 ,'b.')

plt.plot( xgr2, ygr2,'r.')
plt.plot( [0], [0],'g.')

# plt.axis('equal')
plt.grid()
plt.gca().invert_yaxis()
plt.show()

#%%
def diference_grids(cgrx1,cgry1,cgrx2,cgry2, spx1,spy1,spx2,spy2):
    dists = []
    for i in range(-31,30):
        for j in range(-27,28):
            point = [j,i]
            parg1 = np.where( dist(cgrx1,cgry1,point) == 0)[0]
            parg2 = np.where( dist(cgrx2,cgry2,point) == 0)[0]
            
            dx = (spx1[parg1] ) - (spx2[parg2] )  
            dy = (spy1[parg1] ) - (spy2[parg2] )
            dists.append( (dx**2 + dy**2)[0] )
            
    dists = np.array(dists)
    return np.nansum(dists)

def lessq(val):
    rgrx2_ = np.cos(val[0]) * xgr2 - np.sin(val[0]) * ygr2
    rgry2_ = np.sin(val[0]) * xgr2 + np.cos(val[0]) * ygr2
    
    topx = val[1] * rgrx2_ + val[2] * rgry2_ + val[3]
    topy = val[4] * rgrx2_ + val[5] * rgry2_ + val[6]
    bot =  val[7] * rgrx2_ + val[8] * rgry2_ + 1
    
    rgrx2 = topx / bot
    rgry2 = topy / bot
    
    return diference_grids(cgrx1,cgry1,cgrx2,cgry2,  xgr1,ygr1, rgrx2, rgry2 ) 
# (cgrx1, cgry1, rgrx2 * val[1], rgry2 * val[1])
    
t1 = time()

resid = least_squares(lessq, [-0,1,0,0,0,1,0,0,0])

t2 = time()
print(t2-t1)
print(resid.x)

val = resid.x
rgrx2_ = np.cos(val[0]) * xgr2 - np.sin(val[0]) * ygr2
rgry2_ = np.sin(val[0]) * xgr2 + np.cos(val[0]) * ygr2

topx = val[1] * rgrx2_ + val[2] * rgry2_ + val[3]
topy = val[4] * rgrx2_ + val[5] * rgry2_ + val[6]
bot =  val[7] * rgrx2_ + val[8] * rgry2_ + 1

rgrx2 = topx / bot
rgry2 = topy / bot
    
plt.figure()
plt.plot( xgr1, ygr1 ,'b.')

plt.plot( xgr2, ygr2,'r.')
# plt.plot( [0], [0],'g.')

plt.plot( rgrx2, rgry2,'g.')

plt.plot( [0], [0],'y.')

# plt.axis('equal')
plt.grid()
plt.gca().invert_yaxis()
plt.show()



#%%

# diference_grids(cgrx1,cgry1,cgrx2,cgry2, spx1-cx1,spy1-cy1, spx2-cx2, spy2-cy2)

#%%
# =============================================================================
# Video calibration in set up
# =============================================================================
t1 = time()
vid = imageio.get_reader('Documents/Dodecahedro/Calibration/HERO9 BLACK/GX010175.MP4', 'ffmpeg') # 5000 last frame
t2 = time()

print(t2-t1)
#%%
t1 = time()

fima = vid.get_data(1000)[:,:,2]
image = vid.get_data(1000)[180:1010,275:1710,2]
ny,nx = np.shape(vid.get_data(1000)[:,:,2])

imb = image < sfr.mean(image, np.ones((15,15)) )
imo = remove_small_objects(imb, 40000)
imh = remove_small_holes(imo, area_threshold=60)
imk = remove_small_objects( binary_opening(imh, disk(3)), 10000 )
iml = binary_closing( np.pad(remove_small_objects(binary_opening(imk, disk(5)), 10000), 10), disk(10) )[10:-10,10:-10]

mask = binary_erosion(iml, np.ones((10,10)))
imt = image * mask


ddd = sfr.mean(imt, np.ones((20,20)), mask=mask)
eee = imt - ddd*1.
dots = remove_small_objects(eee > threshold_otsu(eee), 6 )

lal = label(dots) 
props = regionprops(lal)

centsy_vid = np.array( [(props[i].centroid)[0] for i in range(len(props)) ] ) + 180
centsx_vid = np.array( [(props[i].centroid)[1] for i in range(len(props)) ] ) + 275

def cent_dist(val):
    di0 = (val[0] - centsx_vid)**2  + (val[1] - centsy_vid)**2
    idi0 = np.argsort( di0 )
    return np.sum(di0[idi0[:8]] ) / di0[idi0[0]]

bounds = ( (int(np.mean(centsx_vid)-100), int(np.mean(centsx_vid)+100)), (int(np.mean(centsy_vid)-100), int(np.mean(centsy_vid)+100)) )
redi = differential_evolution(cent_dist, bounds, integrality = [True,True])

x0 = redi.x
red = least_squares(cent_dist, x0, bounds=((x0[0]-10,x0[1]-10),(x0[0]+10,x0[1]+10)) )

t2 = time()
t2-t1
#%%

t1 = time()

(dot_size, dot_dist) = prep.calc_size_distance(dots)
hor_slope = prep.calc_hor_slope(dots)
ver_slope = prep.calc_ver_slope(dots)

list_hor_lines0 = prep.group_dots_hor_lines(dots, hor_slope, dot_dist,
                                            ratio=0.3, num_dot_miss=10,
                                            accepted_ratio=0.6)
list_ver_lines0 = prep.group_dots_ver_lines(dots, ver_slope, dot_dist,
                                            ratio=0.3, num_dot_miss=10,
                                            accepted_ratio=0.6)
ftu = []
orhgx, orhgy = [], []
hline = np.zeros_like( centsx_vid ) * np.nan
# cenh = [] 

for i in tqdm(range(len(list_hor_lines0))):
    u = np.linspace(0,1500,3000)
    (a,b,c,d),cov = curve_fit(pol3, list_hor_lines0[i][:,1], list_hor_lines0[i][:,0])    
    
    dis = []
    for n in range(len(centsx_vid)):
        point = [centsx_vid[n] - 275, centsy_vid[n] - 180]
        poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )
        dis.append( poop )
    dis = np.array(dis)
        
    fill = dis < 20
    sorting = np.argsort( centsx_vid[fill] )
    orhgx.append( (centsx_vid[fill])[sorting] )
    orhgy.append( (centsy_vid[fill])[sorting] )
    hline[fill] = len(list_hor_lines0) - i
    
    point = [red.x[0] - 275, red.x[1] - 180]
    poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )
    if poop<20: cenh = len(list_hor_lines0) - i
    

orvgx, orvgy = [], []
vline = np.zeros_like( centsx_vid ) * np.nan
# cenv = []

for i in tqdm(range(len(list_ver_lines0))):
    u = np.linspace(0,1500,3000) 
    (a,b,c,d),cov = curve_fit(pol3, list_ver_lines0[i][:,0], list_ver_lines0[i][:,1])    
    
    dis = []
    for n in range(len(centsy_vid)):
        point = [centsx_vid[n] - 275, centsy_vid[n] - 180]
        poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d) , u, point)  )
        dis.append( poop )
    dis = np.array(dis)
    
    fill = dis < 20
    sorting = np.argsort( centsy_vid[fill] )
    orvgx.append( (centsx_vid[fill])[sorting] )
    orvgy.append( (centsy_vid[fill])[sorting] )
    vline[fill] = i
    
    point = [red.x[0] - 275, red.x[1] - 180]
    poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d),u, point)  )
    if poop<20: cenv = i

filtro = vline > 0
order = np.argsort( vline[filtro] + 1j * hline[filtro] )

t2 = time()
print()
print(t2-t1)

#%%
spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]

fin = 100

plt.figure()

plt.imshow(fima, cmap='gray')

plt.plot(nx/2,ny/2,'g.' )
plt.plot( red.x[0], red.x[1], 'm.' )
# plt.plot( centsx_vid, centsy_vid, 'r.', markersize=10 )

plt.plot( spx_vid[:fin], spy_vid[:fin], 'b.', markersize=5 )

plt.show()

vs, hs = vline[filtro][order] - cenv, hline[filtro][order] - cenh

plt.figure()
plt.plot( hs, vs, '.' )
plt.plot( [0], [0], '.' )
plt.grid()
plt.gca().invert_yaxis()
plt.show()

#%%
xmm, ymm = [], []
for n in range(-51,55):
    for m in range(-27,28):
        n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
        n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
                
        ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
            
        xmm.append( np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) ) )
        ymm.append( np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) ) )

xmm, ymm = np.array(xmm), np.array(ymm)

#%%
def poln(x,val,skip=2):
    oop = 0
    for j in range(skip,len(val)):
        oop += val[j] * x**(j-skip)
    return oop


def calib(val):
    dsits = []
    for i in range(len(vs)):    
    # for i in [2000]:
        n,m = vs[i], hs[i]
        n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
        n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
                
        ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
            
        xmm = np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) )
        ymm = np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) )
                
        xpx, ypx = spx_vid[i] - nx/2, spy_vid[i] - ny/2
        # rpx = np.sqrt( (xpx - 0)**2 + (ypx - 0)**2 )
        # # rpx = np.sqrt( (xpx - val[1])**2 + (ypx - val[2])**2 )
        

        # xfir = np.cos(val[0]) * xpx - np.sin(val[0]) * ypx
        # yfir = np.sin(val[0]) * xpx + np.cos(val[0]) * ypx
        # xfir = np.cos(val[0]) * (xpx - val[1]) - np.sin(val[0]) * (ypx - val[2])
        # yfir = np.sin(val[0]) * (xpx - val[1]) + np.cos(val[0]) * (ypx - val[2])

        # # pol = poln(rpx, val, skip=3)        
        # # pol = val[3] + val[4] * rpx**2 #+ val[5] * rpx**4
        # pol = 1 + val[3] * rpx**2 #+ val[5] * rpx**4
        # # xfit, yfit = xpx * pol, ypx * pol
        # xfit, yfit = xfir / pol, yfir / pol
        # dsits.append( (xmm - xfit)**2 + (ymm - yfit)**2 )
        
        xfir = (np.cos(val[0]) * (xmm - 0) - np.sin(val[0]) * (ymm - 0)) * val[1] + red.x[0]-nx/2
        yfir = (np.sin(val[0]) * (xmm - 0) + np.cos(val[0]) * (ymm - 0)) * val[1] + red.x[1]-ny/2
        # rfir = np.sqrt( (xfir-val[3])**2 + (yfir-val[4])**2 )
        rfir = np.sqrt( (xfir)**2 + (yfir)**2 )
        
        # pol = val[1] + val[2] * rfir**2 #+ val[5] * rfir**4
        pol = 1 + val[2] * rfir**2 
        # xfit, yfit = xfir * pol, yfir * pol
        xfit, yfit = (xfir) * pol, (yfir) * pol
        dsits.append( (xpx - xfit)**2 + (ypx - yfit)**2 )
   
    return np.nansum(dsits)

t1 = time()
# les = least_squares(calib, [1.807, red.x[0]-nx/2 , red.x[1]-ny/2, 1, 0.001])
les = least_squares(calib, [-1.85, 2.512, -3e-7])
t2 = time()
print(t2-t1)
print(les.x)
les
#%%
def clibrated(val):
    xpx, ypx = spx_vid - nx/2, spy_vid - ny/2

    # correcting on distorted image
    rpx = np.sqrt( (xpx)**2 + (ypx)**2 )
    pol = val[2] + val[3] * rpx**2
    xfi, yfi = xpx * pol, ypx * pol
    
    xfir = (np.cos(val[0]) * (xfi - (red.x[0]-nx/2) ) - np.sin(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
    yfir = (np.sin(val[0]) * (xfi - (red.x[0]-nx/2) ) + np.cos(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
    rfir = np.sqrt( (xfir + (red.x[0]-nx/2))**2 + (yfir + (red.x[0]-nx/2))**2 )
    xfit, yfit = xfir,yfir
    
    # # pol = poln(rpx, val, skip=3)        
    # # pol = val[3] + val[4] * rpx**2
    # pol = 1 + val[3] * rpx**2
    # # xfit, yfit = xpx * pol, ypx * pol
    # xfit, yfit = xfir / pol, yfir / pol

    # xfir = np.cos(val[0]) * (xmm - val[1]) - np.sin(val[0]) * (ymm - val[2])
    # yfir = np.sin(val[0]) * (xmm - val[1]) + np.cos(val[0]) * (ymm - val[2])
    # rfir = np.sqrt( xfir**2 + yfir**2 )

    # # pol = val[3] + val[4] * rfir**2 + val[5] * rfir**4
    # pol = 1 + val[3] * rfir**2 
    # # xfit, yfit = xfir * pol, yfir * pol
    # xfit, yfit = xfir * pol, yfir * pol

    # correcting on non distorted image
    # xfir = (np.cos(val[0]) * (xmm - 0) - np.sin(val[0]) * (ymm - 0)) * val[1] + red.x[0]-nx/2 #val[1]
    # yfir = (np.sin(val[0]) * (xmm - 0) + np.cos(val[0]) * (ymm - 0)) * val[1] + red.x[1]-ny/2 #val[2]
    # # rfir = np.sqrt( (xfir-val[3])**2 + (yfir-val[4])**2 )
    # rfir = np.sqrt( (xfir)**2 + (yfir)**2 )
    
    # # pol = val[1] + val[2] * rfir**2 #+ val[5] * rfir**4
    # pol = 1 + val[2] * rfir**2 
    # # xfit, yfit = xfir * pol, yfir * pol
    # xfit, yfit = (xfir) * pol, (yfir) * pol

    return xfit, yfit, rfir

spr_vid = np.sqrt( (spx_vid-nx/2)**2 + (spy_vid-ny/2)**2  )
# xfit, yfit, rfir = clibrated(les.x)
# cenrx,cenry = 47/2, 73/2
xfit, yfit, rfir = clibrated( [1.87, 1, 1, 4e-4] )

plt.figure()
# plt.plot( xgr1, ygr1 ,'b.')
# plt.plot( [0], [0],'y.')

# plt.plot( xgr2, ygr2,'r.')
# plt.plot( [0], [0],'g.')

# plt.plot( spx_vid - nx/2, spy_vid - ny/2, '.' )
# plt.plot( (spx_vid-nx/2) * poln(spr_vid, les.x,skip=1), (spy_vid-ny/2) * poln(spr_vid, les.x, skip=1), '.' )
# plt.scatter( xfit, yfit, s=10, c='green')
plt.plot( xfit, yfit, 'g.')

# plt.plot( [cenrx], [cenry], 'ro' )
# plt.plot( [red.x[0]-nx/2], [red.x[1]-ny/2], 'ro' )


plt.axis('equal')
plt.grid()
plt.gca().invert_yaxis()
plt.show()


#%%

rrrx, rrry = rotate_point(-1.85 *1, cgrx1 * 12.5, cgry1 * 12.5)
rgr1 = np.sqrt( rrry**2 + rrrx**2 )
pol = 1 + rgr1**2 * 0.0001
dgx = rrrx / pol
dgy = rrry / pol


# plt.figure()
# plt.plot( cgrx1, cgry1, '.' )
# plt.plot( rrrx, rrry, '.' )
# plt.plot( dgx, dgy, '.' )
# plt.show()


fin = 140

plt.figure()
plt.imshow( fima, cmap='gray', extent=(-nx/2,nx/2,ny/2,-ny/2) )
# plt.plot(nx/2,ny/2,'.')
plt.plot(0,0,'.')

plt.plot( spx_vid[:fin] - nx/2 , spy_vid[:fin] - ny/2 , '.' )
plt.plot( rrrx[:fin] + red.x[0]-nx/2, rrry[:fin] + red.x[1]-ny/2, '.' )
plt.plot( rrrx[din1+1] + red.x[0]-nx/2, rrry[din1+1] + red.x[1]-ny/2, 'r.' )
# plt.plot(cgrx1[din1+1], cgry1[din1+1], '.')
plt.show()

#%%
xpx, ypx = spx_vid - nx/2, spy_vid - ny/2
rpr = np.sqrt( xpx**2 + ypx**2 )
n1 = 1
n2 = 1e-4
n3 = 3e-7 

pol = n1 + n2 * rpr + n3 * rpr**2
krx, kry = rotate_point(1.86, (xpx*pol ) , ypx*pol)

plt.figure()
# plt.plot( xpx, ypx, '.' )
plt.plot( xpx * pol, ypx * pol, '.' )
plt.plot( krx, kry, '.' )
plt.gca().invert_yaxis()
plt.show()

#%%
rrrx, rrry = rotate_point(-1.85 *1, xgr1 * 1.5, ygr1 * 1.5)

fin = 20
plt.figure()
plt.plot( spx_vid - nx/2 , spy_vid - ny/2 , '.' )
plt.plot( xgr1, ygr1, '.' )
plt.plot( rrrx, rrry, '.' )

plt.plot( spx_vid[:fin] - nx/2 , spy_vid[:fin] - ny/2, '.'  )
plt.plot( xgr1[:fin], ygr1[:fin], '.' )
plt.plot( rrrx[:fin], rrry[:fin], '.' )
plt.gca().invert_yaxis()
plt.show()


#%%



#%%









#%%



#%%
# =============================================================================
# Other image
# =============================================================================
t1 = time()

fima = vid.get_data(4200)[:,:,2]
image = vid.get_data(4200)[40:,680:1430,2]
ny,nx = np.shape(vid.get_data(4200)[:,:,2])

imb = image < sfr.mean(image, np.ones((15,15)) )
imo = remove_small_objects(imb, 40000)
imh = remove_small_holes(imo, area_threshold=400)
imk = remove_small_objects( binary_opening(imh, disk(3)), 10000 )
iml = binary_closing( np.pad(remove_small_objects(binary_opening(imk, disk(5)), 10000), 10), disk(10) )[10:-10,10:-10]

mask = binary_erosion(iml, np.ones((10,10)))
imt = image * mask


ddd = sfr.mean(imt, np.ones((20,20)), mask=mask)
eee = imt - ddd*1.
dots = remove_small_objects(eee > threshold_otsu(eee), 9 )

lal = label(dots) 
props = regionprops(lal)

centsy_vid = np.array( [(props[i].centroid)[0] for i in range(len(props)) if props[i].area<50 ] ) + 40
centsx_vid = np.array( [(props[i].centroid)[1] for i in range(len(props)) if props[i].area<50 ] ) + 680

def cent_dist(val):
    di0 = (val[0] - centsx_vid)**2  + (val[1] - centsy_vid)**2
    idi0 = np.argsort( di0 )
    return np.sum(di0[idi0[:8]] ) / di0[idi0[0]]

bounds = ( (int(np.mean(centsx_vid)-100), int(np.mean(centsx_vid)+100)), (int(np.mean(centsy_vid)-100), int(np.mean(centsy_vid)+100)) )
redi = differential_evolution(cent_dist, bounds, integrality = [True,True])

x0 = redi.x
red = least_squares(cent_dist, x0, bounds=((x0[0]-10,x0[1]-10),(x0[0]+10,x0[1]+10)) )

t2 = time()
print(t2-t1)
#%%
t1 = time()

(dot_size, dot_dist) = prep.calc_size_distance(dots)
hor_slope = prep.calc_hor_slope(dots)
ver_slope = prep.calc_ver_slope(dots)

list_hor_lines0 = prep.group_dots_hor_lines(dots, hor_slope, dot_dist,
                                            ratio=0.3, num_dot_miss=10,
                                            accepted_ratio=0.6)
list_ver_lines0 = prep.group_dots_ver_lines(dots, ver_slope, dot_dist,
                                            ratio=0.3, num_dot_miss=10,
                                            accepted_ratio=0.6)
ftu = []
orhgx, orhgy = [], []
hline = np.zeros_like( centsx_vid ) * np.nan
# cenh = [] 

for i in tqdm(range(len(list_hor_lines0))):
    u = np.linspace(0,1500,3000)
    (a,b,c,d),cov = curve_fit(pol3, list_hor_lines0[i][:,1], list_hor_lines0[i][:,0])    
    
    dis = []
    for n in range(len(centsx_vid)):
        point = [centsx_vid[n] - 680, centsy_vid[n] - 40]
        poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )
        dis.append( poop )
    dis = np.array(dis)
        
    fill = dis < 20
    sorting = np.argsort( centsx_vid[fill] )
    orhgx.append( (centsx_vid[fill])[sorting] )
    orhgy.append( (centsy_vid[fill])[sorting] )
    hline[fill] = len(list_hor_lines0) - i
    
    point = [red.x[0] - 680, red.x[1] - 40]
    poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )
    if poop<20: cenh = len(list_hor_lines0) - i
    

orvgx, orvgy = [], []
vline = np.zeros_like( centsx_vid ) * np.nan
# cenv = []

for i in tqdm(range(len(list_ver_lines0))):
    u = np.linspace(0,1500,3000) 
    (a,b,c,d),cov = curve_fit(pol3, list_ver_lines0[i][:,0], list_ver_lines0[i][:,1])    
    
    dis = []
    for n in range(len(centsy_vid)):
        point = [centsx_vid[n] - 680, centsy_vid[n] - 40]
        poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d) , u, point)  )
        dis.append( poop )
    dis = np.array(dis)
    
    fill = dis < 20
    sorting = np.argsort( centsy_vid[fill] )
    orvgx.append( (centsx_vid[fill])[sorting] )
    orvgy.append( (centsy_vid[fill])[sorting] )
    vline[fill] = i
    
    point = [red.x[0] - 680, red.x[1] - 40]
    poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d),u, point)  )
    if poop<20: cenv = i

filtro = (hline > 1) * (hline < 101)
order = np.argsort( -hline[filtro] + 1j * vline[filtro] )

t2 = time()
print()
print(t2-t1)
#%%
spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]
vs, hs = cenh - hline[filtro][order] , vline[filtro][order] - cenv


# plt.figure()
# # plt.imshow(fima, cmap='gray')
# plt.imshow(image, cmap='gray')
# # for i in range(len(list_hor_lines0)):
# for i in range(10,11):
#     u = np.linspace(0,800,3000)
#     (a,b,c,d),cov = curve_fit(pol3, list_hor_lines0[i][:,1], list_hor_lines0[i][:,0])    
#     plt.plot( list_hor_lines0[i][:,1] + 0, list_hor_lines0[i][:,0] + 0, '.-' )
#     plt.plot( u, pol3(u, a, b, c, d), '-' )
# plt.show()

fin = 50

plt.figure()
plt.plot( hs, vs, '.' )
plt.plot( hs[:fin], vs[:fin], '.' )
plt.show()

plt.figure()
plt.imshow( gridf1, cmap='gray' )
# plt.plot( xgr1, ygr1, 'b.' )
# plt.plot( xgr2, ygr2, 'b.' )
# plt.plot( xgr1[:fin], ygr1[:fin], 'g.' )
plt.plot( cgrx1[:fin], cgry1[:fin], 'g.' )
# plt.plot( xgr2[:fin], ygr2[:fin], '.' )
plt.show()


plt.figure()
plt.imshow(fima)
plt.plot( spx_vid[:fin], spy_vid[:fin], 'r.')
plt.show()

#%%
xmm, ymm = [], []
for n in range(-51,55):
    for m in range(-27,28):
        n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
        n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
                
        ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
            
        xmm.append( np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) ) )
        ymm.append( np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) ) )

xmm, ymm = np.array(xmm), np.array(ymm)

#%%
def poln(x,val,skip=2):
    oop = 0
    for j in range(skip,len(val)):
        oop += val[j] * x**(j-skip)
    return oop


def calib(val):
    dsits = []
    for i in range(len(vs)):    
    # for i in [2000]:
        n,m = vs[i], hs[i]
        n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
        n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
                
        ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
            
        xmm = np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) )
        ymm = np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) )
                
        xpx, ypx = spx_vid[i] - nx/2, spy_vid[i] - ny/2
        rpx = np.sqrt( (xpx - 0)**2 + (ypx - 0)**2 )

        pol = 1 + val[2] * rpx**2 + val[3] * rpx**4
        xfi, yfi = xpx * pol, ypx * pol
        # rpx = np.sqrt( (xpx - val[1])**2 + (ypx - val[2])**2 )        

        # xfir = np.cos(val[0]) * xpx - np.sin(val[0]) * ypx
        # yfir = np.sin(val[0]) * xpx + np.cos(val[0]) * ypx
        # xfir = np.cos(val[0]) * (xpx - val[1]) - np.sin(val[0]) * (ypx - val[2])
        # yfir = np.sin(val[0]) * (xpx - val[1]) + np.cos(val[0]) * (ypx - val[2])
        xfir = (np.cos(val[0]) * (xfi - (red.x[0]-nx/2) ) - np.sin(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
        yfir = (np.sin(val[0]) * (xfi - (red.x[0]-nx/2) ) + np.cos(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
        # rfir = np.sqrt( (xfir + (red.x[0]-nx/2))**2 + (yfir + (red.x[0]-nx/2))**2 )
        # xfit, yfit = xfir - 0, yfir - 0
        xfit, yfit = xfir - val[4], yfir - val[5]

        # pol = poln(rpx, val, skip=3)        
        # pol = val[3] + val[4] * rpx**2 #+ val[5] * rpx**4
        # pol = 1 + val[3] * rfir**2 #+ val[5] * rpx**4
        # xfit, yfit = xpx * pol, ypx * pol
        # xfit, yfit = xfir / pol, yfir / pol
        dsits.append( (xmm - xfit)**2 + (ymm - yfit)**2 )
        
        # xfir = (np.cos(val[0]) * (xmm - 0) - np.sin(val[0]) * (ymm - 0)) * val[1] + red.x[0]-nx/2
        # yfir = (np.sin(val[0]) * (xmm - 0) + np.cos(val[0]) * (ymm - 0)) * val[1] + red.x[1]-ny/2
        # # rfir = np.sqrt( (xfir-val[3])**2 + (yfir-val[4])**2 )
        # rfir = np.sqrt( (xfir)**2 + (yfir)**2 )
        
        # # pol = val[1] + val[2] * rfir**2 #+ val[5] * rfir**4
        # pol = 1 + val[2] * rfir**2 
        # # xfit, yfit = xfir * pol, yfir * pol
        # xfit, yfit = (xfir) * pol, (yfir) * pol
        # dsits.append( (xpx - xfit)**2 + (ypx - yfit)**2 )
   
    return np.nansum(dsits)

t1 = time()
# les = least_squares(calib, [1.807, red.x[0]-nx/2 , red.x[1]-ny/2, 1, 0.001])
les = least_squares(calib, [0, 0.43, 3.5e-7, 1e-12, 0.33,0.16], bounds=((0,0,0,0,-3,-3),(np.pi/2,5,1e-6,1e-10,3,3)))
# les = least_squares(calib, [0, 0.38, 4.8e-7], bounds=((0,0,0),(1,5,1e-6)))
t2 = time()
print(t2-t1)
print(les.x)
les
#%%
xpx, ypx = spx_vid - nx/2, spy_vid - ny/2
def clibrated(val):

    # correcting on distorted image
    rpx = np.sqrt( (xpx)**2 + (ypx)**2 )
    pol = 1 + val[2] * rpx**2 + val[3] * rpx**4
    xfi, yfi = xpx * pol, ypx * pol
    
    xfir = (np.cos(val[0]) * (xfi - (red.x[0]-nx/2) ) - np.sin(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
    yfir = (np.sin(val[0]) * (xfi - (red.x[0]-nx/2) ) + np.cos(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
    rfir = np.sqrt( (xfir + (red.x[0]-nx/2))**2 + (yfir + (red.x[0]-nx/2))**2 )
    # xfit, yfit = xfir - 0, yfir - 0
    xfit, yfit = xfir - val[4], yfir - val[5]
    
    # # pol = poln(rpx, val, skip=3)        
    # # pol = val[3] + val[4] * rpx**2
    # pol = 1 + val[3] * rpx**2
    # # xfit, yfit = xpx * pol, ypx * pol
    # xfit, yfit = xfir / pol, yfir / pol

    # xfir = np.cos(val[0]) * (xmm - val[1]) - np.sin(val[0]) * (ymm - val[2])
    # yfir = np.sin(val[0]) * (xmm - val[1]) + np.cos(val[0]) * (ymm - val[2])
    # rfir = np.sqrt( xfir**2 + yfir**2 )

    # # pol = val[3] + val[4] * rfir**2 + val[5] * rfir**4
    # pol = 1 + val[3] * rfir**2 
    # # xfit, yfit = xfir * pol, yfir * pol
    # xfit, yfit = xfir * pol, yfir * pol

    # correcting on non distorted image
    # xfir = (np.cos(val[0]) * (xmm - 0) - np.sin(val[0]) * (ymm - 0)) * val[1] + red.x[0]-nx/2 #val[1]
    # yfir = (np.sin(val[0]) * (xmm - 0) + np.cos(val[0]) * (ymm - 0)) * val[1] + red.x[1]-ny/2 #val[2]
    # # rfir = np.sqrt( (xfir-val[3])**2 + (yfir-val[4])**2 )
    # rfir = np.sqrt( (xfir)**2 + (yfir)**2 )
    
    # # pol = val[1] + val[2] * rfir**2 #+ val[5] * rfir**4
    # pol = 1 + val[2] * rfir**2 
    # # xfit, yfit = xfir * pol, yfir * pol
    # xfit, yfit = (xfir) * pol, (yfir) * pol

    return xfit, yfit, rfir

spr_vid = np.sqrt( (spx_vid-nx/2)**2 + (spy_vid-ny/2)**2  )
# xfit, yfit, rfir = clibrated(les.x)
# cenrx,cenry = 47/2, 73/2
xfit, yfit, rfir = clibrated( [0, 0.4, 4.1e-7, 5e-13, 0.5, 0.1] )

plt.figure()
plt.plot( xgr1, ygr1 ,'b.')
# plt.plot( [0], [0],'y.')

plt.plot( xgr2, ygr2,'r.')
# plt.plot( [0], [0],'g.')

# plt.plot( spx_vid - nx/2, spy_vid - ny/2, '.' )
# plt.plot( (spx_vid-nx/2) * poln(spr_vid, les.x,skip=1), (spy_vid-ny/2) * poln(spr_vid, les.x, skip=1), '.' )
# plt.scatter( xfit, yfit, s=10, c='green')
plt.plot( xfit, yfit, 'g.')

# plt.plot( [cenrx], [cenry], 'ro' )
# plt.plot( [red.x[0]-nx/2], [red.x[1]-ny/2], 'ro' )


plt.axis('equal')
plt.grid()
plt.gca().invert_yaxis()
plt.show()







#%%
ns = 20
v3 = np.logspace(-10,-5,ns)
v2 = np.linspace(0.1,1,ns)
sss, sv2, sv3 = [], [],[]
for k in tqdm(range(ns)):
    for j in range(ns):
        val = [0, v2[j], v3[k]]
        sv2.append(v2[j])
        sv3.append(v3[k])
        
        dsits = []
        for i in range(len(vs)):    
        # for i in [2000]:
            n,m = vs[i], hs[i]
            n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
            n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
                    
            ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
                
            xmm = np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) )
            ymm = np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) )
                    
            xpx, ypx = spx_vid[i] - nx/2, spy_vid[i] - ny/2
            rpx = np.sqrt( (xpx - 0)**2 + (ypx - 0)**2 )
        
            pol = 1 + val[2] * rpx**2
            xfi, yfi = xpx * pol, ypx * pol
            # rpx = np.sqrt( (xpx - val[1])**2 + (ypx - val[2])**2 )        
        
            # xfir = np.cos(val[0]) * xpx - np.sin(val[0]) * ypx
            # yfir = np.sin(val[0]) * xpx + np.cos(val[0]) * ypx
            # xfir = np.cos(val[0]) * (xpx - val[1]) - np.sin(val[0]) * (ypx - val[2])
            # yfir = np.sin(val[0]) * (xpx - val[1]) + np.cos(val[0]) * (ypx - val[2])
            xfir = (np.cos(val[0]) * (xfi - (red.x[0]-nx/2) ) - np.sin(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
            yfir = (np.sin(val[0]) * (xfi - (red.x[0]-nx/2) ) + np.cos(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
            # rfir = np.sqrt( (xfir + (red.x[0]-nx/2))**2 + (yfir + (red.x[0]-nx/2))**2 )
            xfit, yfit = xfir - 0, yfir - 0
            # xfit, yfit = xfir - val[3], yfir - val[4]
        
            # pol = poln(rpx, val, skip=3)        
            # pol = val[3] + val[4] * rpx**2 #+ val[5] * rpx**4
            # pol = 1 + val[3] * rfir**2 #+ val[5] * rpx**4
            # xfit, yfit = xpx * pol, ypx * pol
            # xfit, yfit = xfir / pol, yfir / pol
            dsits.append( (xmm - xfit)**2 + (ymm - yfit)**2 )
        
        sss.append( np.nansum(dsits) )
    
    # plt.figure()
    # plt.plot( xgr1, ygr1 ,'b.')
    # plt.plot( xgr2, ygr2,'r.')
    # plt.plot( xfit, yfit, 'g.')

    # plt.title( np.nansum(dsits) )
    # plt.axis('equal')
    # plt.grid()
    # plt.gca().invert_yaxis()
    # plt.show()

#%%
plt.figure()
# ax = plt.axes(projection='3d')
# plt.plot( v3, sss, '.-' )
# plt.scatter( sv3, sv2, c=np.log(sss), s=np.log(sss) )
# ax.scatter( np.log(sv3), sv2, np.log(sss) )
plt.plot( np.log(sss), '.-' )
# plt.xscale('log')
plt.show()

n = 0
val = [0, v2[n], v3[n]]
# val = [0, 0.4, 3.6e-9]
xpx, ypx = spx_vid - nx/2, spy_vid - ny/2
rpx = np.sqrt( (xpx - 0)**2 + (ypx - 0)**2 )

pol = 1 + val[2] * rpx**2
xfi, yfi = xpx * pol, ypx * pol
xfir = (np.cos(val[0]) * (xfi - (red.x[0]-nx/2) ) - np.sin(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
yfir = (np.sin(val[0]) * (xfi - (red.x[0]-nx/2) ) + np.cos(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
xfit, yfit = xfir - 0, yfir - 0

plt.figure()
plt.plot( xgr1, ygr1, 'b.')
plt.plot( xgr2, ygr2, 'r.')
plt.plot( xfit, yfit, 'g.')

plt.title( np.log(sss[n]) )
plt.axis('equal')
plt.grid()
plt.gca().invert_yaxis()
plt.show()


#%%
val = [0, 0.4, 4e-7]
plt.figure()

plt.plot( xgr1, ygr1, 'b.', markersize=5)
plt.plot( xgr2, ygr2, 'r.', markersize=5)

xpx, ypx = spx_vid - nx/2, spy_vid - ny/2
rpx = np.sqrt( (xpx - 0)**2 + (ypx - 0)**2 )

pol = 1 + val[2] * rpx**2
xfi, yfi = xpx * pol, ypx * pol
xfir = (np.cos(val[0]) * (xfi - (red.x[0]-nx/2) ) - np.sin(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
yfir = (np.sin(val[0]) * (xfi - (red.x[0]-nx/2) ) + np.cos(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
xfit, yfit = xfir - 0, yfir - 0
plt.plot( xfit, yfit, 'g.', markersize=5)

# for i in range(len(vs)):    
for i in range(59):    
    n,m = vs[i], hs[i]
    
    n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
    n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
            
    ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
        
    xmm = np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) )
    ymm = np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) )
            
    xpx, ypx = spx_vid[i] - nx/2, spy_vid[i] - ny/2
    rpx = np.sqrt( (xpx - 0)**2 + (ypx - 0)**2 )

    pol = 1 + val[2] * rpx**2
    xfi, yfi = xpx * pol, ypx * pol

    xfir = (np.cos(val[0]) * (xfi - (red.x[0]-nx/2) ) - np.sin(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
    yfir = (np.sin(val[0]) * (xfi - (red.x[0]-nx/2) ) + np.cos(val[0]) * (yfi - (red.x[1]-ny/2) ) ) * val[1] 
    # rfir = np.sqrt( (xfir + (red.x[0]-nx/2))**2 + (yfir + (red.x[0]-nx/2))**2 )
    xfit, yfit = xfir - 0, yfir - 0

    plt.plot( [xmm,xfit], [ymm,yfit], '-', markersize=2 )
    
    # dsits.append( (xmm - xfit)**2 + (ymm - yfit)**2 )

plt.axis('equal')
plt.grid()
plt.gca().invert_yaxis()
plt.show()

#%%

fin = 1 
plt.figure()
# plt.imshow( gridf1 ,cmap='gray' )
# plt.plot( red.x[0], red.x[1], 'g.' )
plt.plot( 0, 0, 'g.')
plt.plot( xgr1, ygr1, 'r.')
plt.plot( xgr1[0], ygr1[0], 'b.')
plt.grid()
plt.gca().invert_yaxis()
plt.show()

plt.figure()
plt.imshow( fima, cmap='gray' )
plt.plot( red.x[0], red.x[1], 'g.' )
plt.plot( spx_vid, spy_vid, 'r.' )
plt.plot( spx_vid[0], spy_vid[0], 'b.' )
# plt.grid()
# plt.gca().invert_yaxis()
plt.show()



print( cgrx1[0], cgry1[0] )
print( hs[0], vs[0])
#%%


def calib(val):
    dsits = []
    for i in range(len(vs)):    
    # for i in [2000]:
        n,m = vs[i], hs[i]
        n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
        n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
                
        ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
            
        xmm = np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) )
        ymm = np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) )
                
        xpx, ypx = spx_vid[i] - nx/2, spy_vid[i] - ny/2
        xpx, ypx = xpx - (red.x[0]-nx/2), ypx - (red.x[1]-ny/2)

        xtop = val[0]*xpx + val[1]*ypx + val[2]
        ytop = val[3]*xpx + val[4]*ypx + val[5]
        bot =  val[6]*xpx + val[7]*ypx + 1 

        xfit, yfit = xtop/bot, ytop/bot

        dsits.append( (xmm - xfit)**2 + (ymm - yfit)**2 )
   
    dsits = np.array(dsits)
    return dsits[~np.isnan(dsits)]


def calib2(val):
    dsits = []
    for i in range(len(vs)):    
    # for i in [2000]:
        n,m = vs[i], hs[i]
        n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
        n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
                
        ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
            
        xmm = np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) )
        ymm = np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) )
                
        xpx, ypx = spx_vid[i] - nx/2, spy_vid[i] - ny/2
        xpx, ypx = xpx - (red.x[0]-nx/2), ypx - (red.x[1]-ny/2)

        xtop = val[0]*xpx + val[1]*ypx + val[2] +  val[8] * xpx**2 +  val[9] * ypx**2 + val[10] * ypx*xpx
        ytop = val[3]*xpx + val[4]*ypx + val[5] + val[11] * xpx**2 + val[12] * ypx**2 + val[13] * ypx*xpx
        bot =  val[6]*xpx + val[7]*ypx + 1      + val[14] * xpx**2 + val[15] * ypx**2 + val[16] * ypx*xpx 

        xfit, yfit = xtop/bot, ytop/bot

        dsits.append( (xmm - xfit)**2 + (ymm - yfit)**2 )
   
    dsits = np.array(dsits)
    return dsits[~np.isnan(dsits)]

t1 = time()
# les = least_squares(calib, [0.5, 0, 0, 0, 0.5, 0, 0, 0 ], method='lm' )
# les = least_squares(calib, [ 4.22194302e-01,  9.49900565e-05, -2.41526659e-01,  2.14217660e-03, \
#                             4.76037178e-01,  3.01773404e+00, -9.01380321e-05, -5.37006785e-05] , method='lm' )
les = least_squares(calib2, [ 4.22194302e-01,  9.49900565e-05, -2.41526659e-01,  2.14217660e-03, \
                             4.76037178e-01,  3.01773404e+00, -9.01380321e-05, -5.37006785e-05, \
                             0, 0, 0, 0, 0, 0, 0, 0, 0 ] , method='lm' )
t2 = time()
print(t2-t1)
print(les.x)

# [ 4.22110233e-01  1.78845752e-04 -2.41526529e-01  2.11231534e-03
#   4.75578873e-01  3.01773387e+00 -2.03077923e-05  1.23121569e-03
#   2.22657511e-05  1.05299419e-05  5.50156430e-04 -2.51371400e-05
#   6.36101010e-04  3.19155269e-05  1.80791690e-07 -2.49191494e-07
#  -1.30706463e-07]

#%%

xpx, ypx = spx_vid - nx/2, spy_vid - ny/2

def clibrated(val):
    
    xpxc, ypxc = xpx - (red.x[0]-nx/2), ypx - (red.x[1]-ny/2)
    # xpxc, ypxc = xpx - (red.x[0]*0-nx/2), ypx - (red.x[1]*0-ny/2)
    
    xtop = val[0] * xpxc + val[1] * ypxc + val[2] 
    ytop = val[3] * xpxc + val[4] * ypxc + val[5]
    bot  = val[6] * xpxc + val[7] * ypxc + 1
    xfit, yfit = xtop/bot, ytop/bot

    return xfit, yfit

def clibrated2(val):
    
    xpxc, ypxc = xpx - (red.x[0]-nx/2), ypx - (red.x[1]-ny/2)
    # xpxc, ypxc = xpx - (red.x[0]*0-nx/2), ypx - (red.x[1]*0-ny/2)
    
    xtop = val[0] * xpxc + val[1] * ypxc + val[2] +  val[8] * xpx**2 +  val[9] * ypx**2 + val[10] * ypx*xpx
    ytop = val[3] * xpxc + val[4] * ypxc + val[5] + val[11] * xpx**2 + val[12] * ypx**2 + val[13] * ypx*xpx
    bot  = val[6] * xpxc + val[7] * ypxc + 1      + val[14] * xpx**2 + val[15] * ypx**2 + val[16] * ypx*xpx
    xfit, yfit = xtop/bot, ytop/bot

    return xfit, yfit



spr_vid = np.sqrt( (spx_vid-nx/2)**2 + (spy_vid-ny/2)**2  )

# xfit, yfit = clibrated(les.x)
# xfit, yfit = clibrated( [0.5, 0, 0, 0, 0.5, 0, 0, 0 ] )
# xfit, yfit = clibrated( [1, 0, 0, 0, 1, 0, 0, 0 ] )

xfit, yfit = clibrated2(les.x)
# xfit, yfit = clibrated( [0.5, 0, 0, 0, 0.5, 0, 0, 0, \
                            # 0 , 0, 0, 0,   0, 0, 0, 0] )

plt.figure()

plt.plot( xgr1, ygr1 ,'b.')
plt.plot( xgr2, ygr2,'r.')
plt.plot( xfit, yfit, 'g.')

plt.axis('equal')
plt.grid()
plt.gca().invert_yaxis()
plt.show()



# plt.figure()

# plt.imshow( fima, cmap='gray' )
# plt.plot( xfit, yfit, 'g.')

# # plt.axis('equal')
# # plt.grid()
# # plt.gca().invert_yaxis()
# plt.show()




#%%





#%%
# =============================================================================
# Local calibration with Delaunay tesselation
# =============================================================================
t1 = time()
vid = imageio.get_reader('Documents/Dodecahedro/Calibration/HERO9 BLACK/GX010175.MP4', 'ffmpeg') # 5000 last frame
t2 = time()

print(t2-t1)

#%%

def get_dots( image, xdisp, ydisp, hole_size=400, dot_size=9, min_size=40000, object_size=10000 ):

    imb = image < sfr.mean(image, np.ones((15,15)) )
    imo = remove_small_objects(imb, min_size=min_size)
    imh = remove_small_holes(imo, area_threshold=hole_size)
    imk = remove_small_objects( binary_opening(imh, disk(3)), object_size )
    iml = binary_closing( np.pad( remove_small_objects(binary_opening(imk, disk(5)), 10000), 10), disk(10) )[10:-10,10:-10]

    mask = binary_erosion(iml, np.ones((10,10)))
    imt = image * mask


    ddd = sfr.mean(imt, np.ones((20,20)), mask=mask)
    eee = imt - ddd*1.
    dots = remove_small_objects(eee > threshold_otsu(eee), dot_size )

    lal = label(dots) 
    props = regionprops(lal)

    centsy_vid = np.array( [(props[i].centroid)[0] for i in range(len(props)) if props[i].area<50 ] ) + ydisp 
    centsx_vid = np.array( [(props[i].centroid)[1] for i in range(len(props)) if props[i].area<50 ] ) + xdisp

    def cent_dist(val):
        di0 = (val[0] - centsx_vid)**2  + (val[1] - centsy_vid)**2
        idi0 = np.argsort( di0 )
        return np.sum(di0[idi0[:8]] ) / di0[idi0[0]]

    bounds = ( (int(np.mean(centsx_vid)-100), int(np.mean(centsx_vid)+100)), (int(np.mean(centsy_vid)-100), int(np.mean(centsy_vid)+100)) )
    redi = differential_evolution(cent_dist, bounds, integrality = [True,True])

    x0 = redi.x
    red = least_squares(cent_dist, x0, bounds=((x0[0]-10,x0[1]-10),(x0[0]+10,x0[1]+10)) )
    return centsx_vid, centsy_vid, red, dots 

def hv_lines(dots, centsx_vid, centsy_vid, red, xdisp, ydisp, h_difference=True, v_difference=False, slope_ver=0):

    (dot_size, dot_dist) = prep.calc_size_distance(dots)
    hor_slope = prep.calc_hor_slope(dots)
    ver_slope = prep.calc_ver_slope(dots)
    if np.abs(slope_ver) > 0: ver_slope = slope_ver

    list_hor_lines0 = prep.group_dots_hor_lines(dots, hor_slope, dot_dist,
                                                ratio=0.3, num_dot_miss=10,
                                                accepted_ratio=0.6)
    list_ver_lines0 = prep.group_dots_ver_lines(dots, ver_slope, dot_dist,
                                                ratio=0.3, num_dot_miss=10,
                                                accepted_ratio=0.6)
    
    hline = np.zeros_like( centsx_vid ) * np.nan
    for i in tqdm(range(len(list_hor_lines0))):
        u = np.linspace(0,1500,3000)
        (a,b,c,d),cov = curve_fit(pol3, list_hor_lines0[i][:,1], list_hor_lines0[i][:,0])    
        
        dis = []
        for n in range(len(centsx_vid)):
            point = [centsx_vid[n] - xdisp, centsy_vid[n] - ydisp]
            poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )
            dis.append( poop )
        dis = np.array(dis)
            
        fill = dis < 20
        
        point = [red.x[0] - xdisp, red.x[1] - ydisp]
        poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )

        if h_difference:
            hline[fill] = len(list_hor_lines0) - i
            if poop<20: cenh = len(list_hor_lines0) - i
        else:
            hline[fill] = i
            if poop<20: cenh = i
        
    vline = np.zeros_like( centsx_vid ) * np.nan
    for i in tqdm(range(len(list_ver_lines0))):
        u = np.linspace(0,1500,3000) 
        (a,b,c,d),cov = curve_fit(pol3, list_ver_lines0[i][:,0], list_ver_lines0[i][:,1])    
        
        dis = []
        for n in range(len(centsy_vid)):
            point = [centsx_vid[n] - xdisp, centsy_vid[n] - ydisp]
            poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d) , u, point)  )
            dis.append( poop )
        dis = np.array(dis)
        
        fill = dis < 20
        
        point = [red.x[0] - xdisp, red.x[1] - ydisp]
        poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d),u, point)  )

        if v_difference:
            vline[fill] = len(list_ver_lines0) - i
            if poop<20: cenv = len(list_ver_lines0) - i
        else:
            vline[fill] = i
            if poop<20: cenv = i
        
    return vline, hline, cenh, cenv

def real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, nclose=4, sign=1, rot = 0, reverse=False ):
    xgri, ygri = [], []
    for i in tqdm(range(len(vs))):
        n,m = vs[i], hs[i]
        n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
        n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
        
        ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
            
        xmm = np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) )
        ymm = np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) )
        
        xgri.append(xmm)
        ygri.append(ymm)

    grid_points = np.vstack((ygri,xgri)).T
    points = np.vstack( (spy_vid, spx_vid) ).T

    tri = Delaunay(points)

    p = np.array([[ny/2, nx/2]])

    s = tri.find_simplex(p)
    # v = tri.vertices[s]
    v = tri.simplices[s]
    m = tri.transform[s]
    b = np.einsum('ijk,ik->ij', m[:,:2,:2], p-m[:,2,:])
    w = np.c_[b, 1-b.sum(axis=1)]

    pgr = np.sum((grid_points[v[0]].T * w[0]).T, axis=0 )

    close = np.argsort(np.sum((points - p)**2, axis=1))[:nclose]
    ord_close = np.argsort( hs[close] + 1j * vs[close] )
    lress, lresi, angu = [],[], []
    # for i in range(3):
    #     lin = points[close][ord_close][i*4:i*4+4]
    for i in range(2):
        lin = points[close][ord_close][i*2:i*2+2]

        if reverse: lres = linregress( lin[:,1], lin[:,0] )
        else: lres = linregress( lin[:,0], lin[:,1] )

        lress.append(lres[0])
        lresi.append(lres[1])
        angu.append( np.arctan(lres[0]) )

    ang = np.mean(angu) * sign + rot

    txgr, tygr = rotate_point(ang, grid_points[:,1] - pgr[1], grid_points[:,0] - pgr[0])
    return txgr, tygr

#%%
t1 = time()
fima = vid.get_data(650)[:,:,2]
image = vid.get_data(650)[150:,200:1700,2]
ny,nx = np.shape(vid.get_data(4200)[:,:,2])

centsx_vid, centsy_vid, red, dots = get_dots( image, 200, 150, hole_size=400, dot_size=9, min_size=40000, object_size=10000 )

vline, hline, cenh, cenv = hv_lines(dots, centsx_vid, centsy_vid, red, 200, 150, h_difference=True, v_difference=False)
# que hacia h_difference?

filtro = (vline > 1) * (hline > 0)
order = np.argsort( vline[filtro] + 1j * hline[filtro] )

spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]
hs, vs = hline[filtro][order] - cenh , vline[filtro][order] - cenv

txgr, tygr = real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, nclose=4, sign=1, rot = -np.pi/2, reverse=True )

t2 = time()
print(t2-t1)

#%%

xdisp, ydisp = 200, 150
hole_size=400
dot_size=9
min_size=40000
object_size=10000 
    
imb = image < sfr.mean(image, np.ones((15,15)) )
imo = remove_small_objects(imb, min_size=min_size)
imh = remove_small_holes(imo, area_threshold=hole_size)
imk = remove_small_objects( binary_opening(imh, disk(3)), object_size )
iml = binary_closing( np.pad( remove_small_objects(binary_opening(imk, disk(5)), 10000), 10), disk(10) )[10:-10,10:-10]

mask = binary_erosion(iml, np.ones((10,10)))
imt = image * mask


ddd = sfr.mean(imt, np.ones((20,20)), mask=mask)
eee = imt - ddd*1.
dots = remove_small_objects(eee > threshold_otsu(eee), dot_size )

lal = label(dots) 
props = regionprops(lal)

centsy_vid = np.array( [(props[i].centroid)[0] for i in range(len(props)) if props[i].area<50 ] ) + ydisp 
centsx_vid = np.array( [(props[i].centroid)[1] for i in range(len(props)) if props[i].area<50 ] ) + xdisp

def cent_dist(val):
    di0 = (val[0] - centsx_vid)**2  + (val[1] - centsy_vid)**2
    idi0 = np.argsort( di0 )
    return np.sum(di0[idi0[:8]] ) / di0[idi0[0]]

bounds = ( (int(np.mean(centsx_vid)-100), int(np.mean(centsx_vid)+100)), (int(np.mean(centsy_vid)-100), int(np.mean(centsy_vid)+100)) )
redi = differential_evolution(cent_dist, bounds, integrality = [True,True])

x0 = redi.x
red = least_squares(cent_dist, x0, bounds=((x0[0]-10,x0[1]-10),(x0[0]+10,x0[1]+10)) )

#%%

xdisp, ydisp = 200, 150
h_difference=True
v_difference=False
slope_ver=0

(dot_size, dot_dist) = prep.calc_size_distance(dots)
hor_slope = prep.calc_hor_slope(dots)
ver_slope = prep.calc_ver_slope(dots)
if np.abs(slope_ver) > 0: ver_slope = slope_ver

list_hor_lines0 = prep.group_dots_hor_lines(dots, hor_slope, dot_dist,
                                            ratio=0.3, num_dot_miss=10,
                                            accepted_ratio=0.6)
list_ver_lines0 = prep.group_dots_ver_lines(dots, ver_slope, dot_dist,
                                            ratio=0.3, num_dot_miss=10,
                                            accepted_ratio=0.6)

plt.figure()
plt.imshow( fima )

hline = np.zeros_like( centsx_vid ) * np.nan
for i in tqdm(range(len(list_hor_lines0))):
    u = np.linspace(0,1500,3000)
    (a,b,c,d),cov = curve_fit(pol3, list_hor_lines0[i][:,1], list_hor_lines0[i][:,0])    
    
    plt.plot( list_hor_lines0[i][:,1] + xdisp, list_hor_lines0[i][:,0] + ydisp, '-' )
plt.show()
    
    # dis = []
    # for n in range(len(centsx_vid)):
    #     point = [centsx_vid[n] - xdisp, centsy_vid[n] - ydisp]
    #     poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )
    #     dis.append( poop )
    # dis = np.array(dis)
        
    # fill = dis < 20
    
    # point = [red.x[0] - xdisp, red.x[1] - ydisp]
    # poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )

    # if h_difference:
    #     hline[fill] = len(list_hor_lines0) - i
    #     if poop<20: cenh = len(list_hor_lines0) - i
    # else:
    #     hline[fill] = i
    #     if poop<20: cenh = i
    
plt.figure()
plt.imshow( fima )

vline = np.zeros_like( centsx_vid ) * np.nan
for i in tqdm(range(len(list_ver_lines0))):
    u = np.linspace(0,1500,3000) 
    (a,b,c,d),cov = curve_fit(pol3, list_ver_lines0[i][:,0], list_ver_lines0[i][:,1])

    plt.plot( list_ver_lines0[i][:,1] + xdisp, list_ver_lines0[i][:,0] + ydisp, '-' )
plt.show()    
    
    # dis = []
    # for n in range(len(centsy_vid)):
    #     point = [centsx_vid[n] - xdisp, centsy_vid[n] - ydisp]
    #     poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d) , u, point)  )
    #     dis.append( poop )
    # dis = np.array(dis)
    
    # fill = dis < 20
    
    # point = [red.x[0] - xdisp, red.x[1] - ydisp]
    # poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d),u, point)  )

    # if v_difference:
    #     vline[fill] = len(list_ver_lines0) - i
    #     if poop<20: cenv = len(list_ver_lines0) - i
    # else:
    #     vline[fill] = i
    #     if poop<20: cenv = i


#%%
t1 = time()
xgri, ygri = [], []
for i in tqdm(range(len(vs))):
    n,m = vs[i], hs[i]
    n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
    n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
    
    ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
        
    xmm = np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) )
    ymm = np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) )
    
    xgri.append(xmm)
    ygri.append(ymm)

grid_points = np.vstack((ygri,xgri)).T
points = np.vstack( (spy_vid, spx_vid) ).T

tri = Delaunay(points)

p = np.array([[ny/2, nx/2]])

s = tri.find_simplex(p)
# v = tri.vertices[s]
v = tri.simplices[s]
m = tri.transform[s]
b = np.einsum('ijk,ik->ij', m[:,:2,:2], p-m[:,2,:])
w = np.c_[b, 1-b.sum(axis=1)]

pgr = np.sum((grid_points[v[0]].T * w[0]).T, axis=0 )

close = np.argsort(np.sum((points - p)**2, axis=1))[:12]
ord_close = np.argsort( hs[close] + 1j * vs[close] )
lress, lresi, angu = [],[], []
for i in range(3):
    lin = points[close][ord_close][i*4:i*4+4]

    # lres = linregress( lin[:,0], lin[:,1] )
    lres = linregress( lin[:,1], lin[:,0] )

    lress.append(lres[0])
    lresi.append(lres[1])
    angu.append( np.arctan(lres[0]) )

ang = np.mean(angu)  - np.pi/2
print(ang)

txgr, tygr = rotate_point(ang, grid_points[:,1] - pgr[1], grid_points[:,0] - pgr[0])

t2 = time()
print(t2-t1)

#%%
fin = 550

plt.figure()
plt.imshow( fima, cmap='gray')
# plt.plot( centsx_vid, centsy_vid, '.' )
# plt.plot( red.x[0], red.x[1], '.')
# plt.plot( nx/2, ny/2, '.')


plt.plot( spx_vid, spy_vid, '.' )
plt.plot( spx_vid[:fin], spy_vid[:fin], '.' )

# plt.triplot( points[:,1], points[:,0], tri.simplices )
# plt.triplot( points[:,1], points[:,0], v, 'm-' )
# plt.plot( points[:,1], points[:,0], '.' )
# plt.plot( points[v[0]][:,1], points[v[0]][:,0], 'm.' )
# plt.plot( points[close][:,1], points[close][:,0], 'k.' )
# plt.plot( p[:,1], p[:,0], '.' )

plt.show()




plt.figure()
# plt.imshow( gridf1, cmap='gray' )
plt.plot(xgri, ygri, '.')
plt.plot(xgri[:fin], ygri[:fin], '.')
plt.plot( txgr, tygr, '.' )

plt.axis('equal')
plt.gca().invert_yaxis()
plt.show()

#%%






#%%












#%%












#%%
# =============================================================================
# Full calibration
# =============================================================================
t1 = time()
vid = imageio.get_reader('Documents/Dodecahedro/Calibration/HERO9 BLACK/GX010175.MP4', 'ffmpeg') # 5000 last frame
t2 = time()

print(t2-t1)

#%%

def get_dots( image, xdisp, ydisp, hole_size=400, dot_size=9, min_size=40000, object_size=10000 ):

    imb = image < sfr.mean(image, np.ones((15,15)) )
    imo = remove_small_objects(imb, min_size=min_size)
    imh = remove_small_holes(imo, area_threshold=hole_size)
    imk = remove_small_objects( binary_opening(imh, disk(3)), object_size )
    iml = binary_closing( np.pad( remove_small_objects(binary_opening(imk, disk(5)), 10000), 10), disk(10) )[10:-10,10:-10]

    mask = binary_erosion(iml, np.ones((10,10)))
    imt = image * mask


    ddd = sfr.mean(imt, np.ones((20,20)), mask=mask)
    eee = imt - ddd*1.
    dots = remove_small_objects(eee > threshold_otsu(eee), dot_size )

    lal = label(dots) 
    props = regionprops(lal)

    centsy_vid = np.array( [(props[i].centroid)[0] for i in range(len(props)) if props[i].area<50 ] ) + ydisp 
    centsx_vid = np.array( [(props[i].centroid)[1] for i in range(len(props)) if props[i].area<50 ] ) + xdisp

    def cent_dist(val):
        di0 = (val[0] - centsx_vid)**2  + (val[1] - centsy_vid)**2
        idi0 = np.argsort( di0 )
        return np.sum(di0[idi0[:8]] ) / di0[idi0[0]]

    bounds = ( (int(np.mean(centsx_vid)-100), int(np.mean(centsx_vid)+100)), (int(np.mean(centsy_vid)-100), int(np.mean(centsy_vid)+100)) )
    redi = differential_evolution(cent_dist, bounds, integrality = [True,True])

    x0 = redi.x
    red = least_squares(cent_dist, x0, bounds=((x0[0]-10,x0[1]-10),(x0[0]+10,x0[1]+10)) )
    return centsx_vid, centsy_vid, red, dots 

def hv_lines(dots, centsx_vid, centsy_vid, red, xdisp, ydisp, h_difference=True, v_difference=False, slope_ver=0, slope_hor=0):

    (dot_size, dot_dist) = prep.calc_size_distance(dots)
    hor_slope = prep.calc_hor_slope(dots)
    ver_slope = prep.calc_ver_slope(dots)
    if np.abs(slope_ver) > 0: ver_slope = slope_ver
    if np.abs(slope_hor) > 0: hor_slope = slope_hor

    list_hor_lines0 = prep.group_dots_hor_lines(dots, hor_slope, dot_dist,
                                                ratio=0.3, num_dot_miss=10,
                                                accepted_ratio=0.6)
    list_ver_lines0 = prep.group_dots_ver_lines(dots, ver_slope, dot_dist,
                                                ratio=0.3, num_dot_miss=10,
                                                accepted_ratio=0.6)
    
    hline = np.zeros_like( centsx_vid ) * np.nan
    for i in tqdm(range(len(list_hor_lines0))):
        u = np.linspace(0,1500,3000)
        (a,b,c,d),cov = curve_fit(pol3, list_hor_lines0[i][:,1], list_hor_lines0[i][:,0])    
        
        dis = []
        for n in range(len(centsx_vid)):
            point = [centsx_vid[n] - xdisp, centsy_vid[n] - ydisp]
            poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )
            dis.append( poop )
        dis = np.array(dis)
            
        fill = dis < 20
        
        point = [red.x[0] - xdisp, red.x[1] - ydisp]
        poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )

        if h_difference:
            hline[fill] = len(list_hor_lines0) - i
            if poop<20: cenh = len(list_hor_lines0) - i
        else:
            hline[fill] = i
            if poop<20: cenh = i
        
    vline = np.zeros_like( centsx_vid ) * np.nan
    for i in tqdm(range(len(list_ver_lines0))):
        u = np.linspace(0,1500,3000) 
        (a,b,c,d),cov = curve_fit(pol3, list_ver_lines0[i][:,0], list_ver_lines0[i][:,1])    
        
        dis = []
        for n in range(len(centsy_vid)):
            point = [centsx_vid[n] - xdisp, centsy_vid[n] - ydisp]
            poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d) , u, point)  )
            dis.append( poop )
        dis = np.array(dis)
        
        fill = dis < 20
        
        point = [red.x[0] - xdisp, red.x[1] - ydisp]
        poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d),u, point)  )

        if v_difference:
            vline[fill] = len(list_ver_lines0) - i
            if poop<20: cenv = len(list_ver_lines0) - i
        else:
            vline[fill] = i
            if poop<20: cenv = i
        
    return vline, hline, cenh, cenv

def real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, vs, hs, nclose=4, sign=1, rot=0, reverse=False ):
    xgri, ygri = [], []
    for i in range(len(vs)):
        n,m = vs[i], hs[i]
        n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
        n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
        
        ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
            
        xmm = np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) )
        ymm = np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) )
        
        xgri.append(xmm)
        ygri.append(ymm)

    grid_points = np.vstack((ygri,xgri)).T
    points = np.vstack( (spy_vid, spx_vid) ).T

    tri = Delaunay(points)

    p = np.array([[ny/2, nx/2]])

    s = tri.find_simplex(p)
    # v = tri.vertices[s]
    v = tri.simplices[s]
    m = tri.transform[s]
    b = np.einsum('ijk,ik->ij', m[:,:2,:2], p-m[:,2,:])
    w = np.c_[b, 1-b.sum(axis=1)]

    pgr = np.sum((grid_points[v[0]].T * w[0]).T, axis=0 )

    close = np.argsort(np.sum((points - p)**2, axis=1))[:nclose]
    ord_close = np.argsort( hs[close] + 1j * vs[close] )
    lress, lresi, angu = [],[], []
    
    # for i in range(3):
    #     lin = points[close][ord_close][i*4:i*4+4]
    for i in range(2):
        lin = points[close][ord_close][i*2:i*2+2]

        if reverse: lres = linregress( lin[:,1], lin[:,0] )
        else: lres = linregress( lin[:,0], lin[:,1] )
        lress.append(lres[0])
        lresi.append(lres[1])
        angu.append( np.arctan(lres[0]) )

    ang = np.mean(angu) * sign + rot

    txgr, tygr = rotate_point(ang, grid_points[:,1] - pgr[1], grid_points[:,0] - pgr[0])
    return txgr, tygr

#%%
Image_points = np.empty((0,2))
Real_pos = np.empty((0,2))

t1 = time()
fima = vid.get_data(4200)[:,:,2]
image = vid.get_data(4200)[40:,680:1430,2]
ny,nx = np.shape(vid.get_data(4200)[:,:,2])

centsx_vid, centsy_vid, red, dots = get_dots( image, 680, 40, hole_size=400, dot_size=9, min_size=40000, object_size=10000 )

vline, hline, cenh, cenv = hv_lines(dots, centsx_vid, centsy_vid, red, 680, 40, h_difference=True, v_difference=False)

filtro = (hline > 1) * (hline < 101) * (vline > 0)
order = np.argsort( -hline[filtro] + 1j * vline[filtro] )

spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]
vs, hs = cenh - hline[filtro][order] , vline[filtro][order] - cenv

txgr, tygr = real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, nclose=4, sign=1, rot = 0 )

Image_points = np.vstack( (Image_points, np.vstack((spy_vid, spx_vid)).T) )
Real_pos = np.vstack( (Real_pos, np.vstack((tygr, txgr)).T) )

t2 = time()
print('\n', t2-t1, len(spx_vid))

t1 = time()
fima = vid.get_data(1000)[:,:,2]
image = vid.get_data(1000)[180:1010,275:1710,2]
ny,nx = np.shape(vid.get_data(4200)[:,:,2])

centsx_vid, centsy_vid, red, dots = get_dots( image, 275, 180, hole_size=60, dot_size=6, min_size=40000, object_size=10000 )

vline, hline, cenh, cenv = hv_lines(dots, centsx_vid, centsy_vid, red, 275, 180, h_difference=True, v_difference=False)

filtro = vline>0
order = np.argsort( vline[filtro] + 1j * hline[filtro] )

spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]
hs, vs = hline[filtro][order] - cenh, vline[filtro][order] - cenv

txgr, tygr = real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, nclose=4, sign=1, rot = -np.pi/2, reverse=True )

Image_points = np.vstack( (Image_points, np.vstack((spy_vid, spx_vid)).T) )
Real_pos = np.vstack( (Real_pos, np.vstack((tygr, txgr)).T) )

t2 = time()
print(t2-t1)
print('\n', t2-t1, len(spx_vid))

t1 = time()
fima = vid.get_data(300)[:,:,2]
image = vid.get_data(300)[40:,220:1680,2]
# image = vid.get_data(3000)[100:,220:1680,2]
ny,nx = np.shape(vid.get_data(4200)[:,:,2])

centsx_vid, centsy_vid, red, dots = get_dots( image, 220, 40, hole_size=400, dot_size=9, min_size=40000, object_size=10000 )

vline, hline, cenh, cenv = hv_lines(dots, centsx_vid, centsy_vid, red, 220, 40, h_difference=True, v_difference=False, slope_ver=0.7)

filtro = hline < 98
order = np.argsort( -hline[filtro] + 1j * vline[filtro] )

spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]
vs, hs = cenh - hline[filtro][order] , vline[filtro][order] - cenv

txgr, tygr = real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, nclose=4, sign=-1, rot = 0 )

Image_points = np.vstack( (Image_points, np.vstack((spy_vid, spx_vid)).T[-67:]) )
Real_pos = np.vstack( (Real_pos, np.vstack((tygr, txgr)).T[-67:]) )

t2 = time()
print(t2-t1)
print('\n', t2-t1, len(spx_vid))

t1 = time()
fima = vid.get_data(650)[:,:,2]
image = vid.get_data(650)[150:,200:1700,2]
ny,nx = np.shape(vid.get_data(4200)[:,:,2])

centsx_vid, centsy_vid, red, dots = get_dots( image, 200, 150, hole_size=400, dot_size=9, min_size=40000, object_size=10000 )

vline, hline, cenh, cenv = hv_lines(dots, centsx_vid, centsy_vid, red, 200, 150, h_difference=True, v_difference=False)

filtro = (vline > 1) * (hline > 0)
order = np.argsort( vline[filtro] + 1j * hline[filtro] )

spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]
hs, vs = hline[filtro][order] - cenh , vline[filtro][order] - cenv

txgr, tygr = real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, nclose=4, sign=1, rot = -np.pi/2, reverse=True )

Image_points = np.vstack( (Image_points, np.vstack((spy_vid, spx_vid)).T) )
Real_pos = np.vstack( (Real_pos, np.vstack((tygr, txgr)).T) )

t2 = time()
print(t2-t1)
print('\n', t2-t1, len(spx_vid))


#%%


plt.figure()
plt.imshow( fima, cmap='gray' )
# plt.plot( Image_points[:5343,1], Image_points[:5343,0], '.')
# plt.plot( Image_points[5343:5343+5771,1], Image_points[5343:5343+5771,0], '.')
# plt.plot( Image_points[5343+5771:5343+5771+5140,1], Image_points[5343+5771:5343+5771+5140,0], '.')
# plt.plot( Image_points[5343+5771+5140:5343+5771+5140+5664,1], Image_points[5343+5771+5140:5343+5771+5140+5664,0], '.')
plt.show()

plt.figure()
plt.plot( Real_pos[:5343,1], Real_pos[:5343,0], '.')
plt.plot( Real_pos[5343:5343+5771,1], Real_pos[5343:5343+5771,0], '.')
plt.plot( Real_pos[5343+5771:5343+5771+5140,1], Real_pos[5343+5771:5343+5771+5140,0], '.')
plt.plot( Real_pos[5343+5771+5140:5343+5771+5140+5664,1], Real_pos[5343+5771+5140:5343+5771+5140+5664,0], '.')
plt.axis('equal')
plt.gca().invert_yaxis()
plt.show()

tri = Delaunay(Image_points)
plt.figure()
# plt.imshow( fima ) 

plt.triplot(Image_points[:,1], Image_points[:,0], tri.simplices)
plt.plot( Image_points[:,1], Image_points[:,0], '.')

# plt.triplot(Real_pos[:,1], Real_pos[:,0], tri.simplices)
# plt.plot( Real_pos[:,1], Real_pos[:,0], '.')

plt.axis('equal')
plt.gca().invert_yaxis()
plt.show()

#%%
from concave_hull import concave_hull #, concave_hull_indexes

points = concave_hull( Image_points )

plt.figure()
plt.plot( Image_points[:,1], Image_points[:,0], '.' )
plt.plot( points[:,1], points[:,0], 'r-' )

plt.axis('equal')
plt.gca().invert_yaxis()
plt.show()

#%%

plt.figure()
plt.imshow( vid.get_data(300)[:,:,2] )
plt.show()
plt.figure()
plt.imshow( vid.get_data(650)[:,:,2] )
plt.show()
plt.figure()
plt.imshow( vid.get_data(1000)[:,:,2] )
plt.show()

#%%
# from scipy.spatial import ConvexHull #, convex_hull_plot_2d
# from concave_hull import concave_hull, concave_hull_indexes
import shapely.ops as shops
import shapely.geometry as geometry
import math

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(shops.polygonize(m))
    return shops.cascaded_union(triangles), edge_points

#%%

from matplotlib.collections import LineCollection

for i in range(9):
    alpha = (i+1)*.1
    concave_hull, edge_points = alpha_shape(Image_points, alpha=alpha)

    #print concave_hull
    lines = LineCollection(edge_points)
    plt.figure(figsize=(10,10))
    plt.title('Alpha={0} Delaunay triangulation'.format(alpha))
    plt.gca().add_collection(lines)
    delaunay_points = np.array([point.coords[0] for point in Image_points])
    plt.plot(delaunay_points[:,0], delaunay_points[:,1], 'o', hold=1, color='#f16824')

    # _ = plot_polygon(concave_hull)
    # _ = plt.plot(x,y,'o', color='#f16824')
    
#%%

# hull = ConvexHull(Image_points, qhull_options='Qz')
indeces = concave_hull( Image_points, concavity=2, length_threshold=40 )

plt.figure()
# plt.imshow( fima )
# plt.plot( Image_points[:,1], Image_points[:,0], '.')

plt.plot(Image_points[:,0], Image_points[:,1], 'o')

plt.plot( indeces[:,0], indeces[:,1], '.-'  )

# for simplex in hull.simplices:
#     plt.plot(Image_points[simplex, 0], Image_points[simplex, 1], 'k-')

plt.axis('equal')
plt.gca().invert_yaxis()
plt.show()

#%%




#%%
# =============================================================================
# New calibration
# =============================================================================
def recog_dots(image, xdisp, ydisp, hole_size= 50, dot_size= 3, min_size= 40000, object_size= 10000, max_ar=60, min_ar=5):
    imb = image < sfr.mean(image, np.ones((15,15)) )
    imo = remove_small_objects(imb, min_size=min_size)
    imh = remove_small_holes(imo, area_threshold=hole_size)
    imk = remove_small_objects( binary_opening(imh, disk(3)), object_size )
    iml = binary_closing( np.pad( remove_small_objects(binary_opening(imk, disk(5)), 10000), 10), disk(10) )[10:-10,10:-10]
    
    mask = binary_erosion(iml, np.ones((10,10)))
    imt = image * mask
    
    ddd = sfr.mean(imt, np.ones((20,20)), mask=mask)
    eee = imt - ddd*1.
    normee = eee / np.max(np.abs(eee))
    pokk = sfr.mean(normee, disk(1) ) #, mask=mask)
    # dots = remove_small_objects(normee > threshold_otsu(normee), dot_size )
    # dots = remove_small_objects(normee > threshold_yen(normee), dot_size )
    dots = (gaussian(pokk,1) - gaussian(pokk,10)) > 0
    
    lal = label(dots, connectivity=1) 
    props = regionprops(lal)
    
    centsy_vid = np.array( [(props[i].centroid)[0] for i in range(len(props)) if (props[i].area<max_ar and props[i].area>min_ar ) ] ) + ydisp 
    centsx_vid = np.array( [(props[i].centroid)[1] for i in range(len(props)) if (props[i].area<max_ar and props[i].area>min_ar ) ] ) + xdisp
    
    def cent_dist(val):
        di0 = (val[0] - centsx_vid)**2  + (val[1] - centsy_vid)**2
        idi0 = np.argsort( di0 )
        return np.sum(di0[idi0[:8]] ) / di0[idi0[0]]
    
    bounds = ( (int(np.mean(centsx_vid)-100), int(np.mean(centsx_vid)+100)), (int(np.mean(centsy_vid)-100), int(np.mean(centsy_vid)+100)) )
    redi = differential_evolution(cent_dist, bounds, integrality = [True,True])
    
    x0 = redi.x
    red = least_squares(cent_dist, x0, bounds=((x0[0]-10,x0[1]-10),(x0[0]+10,x0[1]+10)) )
    return centsx_vid, centsy_vid, red, dots


def hv_lines(dots, centsx_vid, centsy_vid, red, xdisp, ydisp, h_difference=True, v_difference=False, slope_ver=0, slope_hor=0, distan=20):

    (dot_size, dot_dist) = prep.calc_size_distance(dots)
    hor_slope = prep.calc_hor_slope(dots)
    ver_slope = prep.calc_ver_slope(dots)
    if np.abs(slope_ver) > 0: ver_slope = slope_ver
    if np.abs(slope_hor) > 0: hor_slope = slope_hor

    list_hor_lines0 = prep.group_dots_hor_lines(dots, hor_slope, dot_dist,
                                                ratio=0.3, num_dot_miss=10,
                                                accepted_ratio=0.6)
    list_ver_lines0 = prep.group_dots_ver_lines(dots, ver_slope, dot_dist,
                                                ratio=0.3, num_dot_miss=10,
                                                accepted_ratio=0.6)
    
    hline = np.zeros_like( centsx_vid ) * np.nan
    for i in tqdm(range(len(list_hor_lines0))):
        u = np.linspace(0,2500,3000)
        (a,b,c,d),cov = curve_fit(pol3, list_hor_lines0[i][:,1], list_hor_lines0[i][:,0])    
        
        dis = []
        for n in range(len(centsx_vid)):
            point = [centsx_vid[n] - xdisp, centsy_vid[n] - ydisp]
            poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )
            dis.append( poop )
        dis = np.array(dis)
            
        fill = dis < distan
        
        point = [red.x[0] - xdisp, red.x[1] - ydisp]
        poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )

        if h_difference:
            hline[fill] = len(list_hor_lines0) - i
            if poop<20: cenh = len(list_hor_lines0) - i
        else:
            hline[fill] = i
            if poop<20: cenh = i
        
    vline = np.zeros_like( centsx_vid ) * np.nan
    for i in tqdm(range(len(list_ver_lines0))):
        u = np.linspace(0,2500,3000) 
        (a,b,c,d),cov = curve_fit(pol3, list_ver_lines0[i][:,0], list_ver_lines0[i][:,1])    
        
        dis = []
        for n in range(len(centsy_vid)):
            point = [centsx_vid[n] - xdisp, centsy_vid[n] - ydisp]
            poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d) , u, point)  )
            dis.append( poop )
        dis = np.array(dis)
        
        fill = dis < distan
        
        point = [red.x[0] - xdisp, red.x[1] - ydisp]
        poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d),u, point)  )

        if v_difference:
            vline[fill] = len(list_ver_lines0) - i
            if poop<20: cenv = len(list_ver_lines0) - i
        else:
            vline[fill] = i
            if poop<20: cenv = i
        
    return vline, hline, cenh, cenv

def real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, vs, hs, nclose=4, sign=1, rot=0, reverse=False ):
    xgri, ygri = [], []
    for i in range(len(vs)):
        n,m = vs[i], hs[i]
        n1,m1 = np.where(cgry1 == n)[0], np.where(cgrx1 == m)[0]
        n2,m2 = np.where(cgry2 == n)[0], np.where(cgrx2 == m)[0]
        
        ind1, ind2 = np.intersect1d(n1, m1), np.intersect1d(n2, m2)
            
        xmm = np.mean( np.concatenate((xgr1[ind1], xgr2[ind2])) )
        ymm = np.mean( np.concatenate((ygr1[ind1], ygr2[ind2])) )
        
        xgri.append(xmm)
        ygri.append(ymm)

    grid_points = np.vstack((ygri,xgri)).T
    points = np.vstack( (spy_vid, spx_vid) ).T

    tri = Delaunay(points)

    p = np.array([[ny/2, nx/2]])

    s = tri.find_simplex(p)
    # v = tri.vertices[s]
    v = tri.simplices[s]
    m = tri.transform[s]
    b = np.einsum('ijk,ik->ij', m[:,:2,:2], p-m[:,2,:])
    w = np.c_[b, 1-b.sum(axis=1)]

    pgr = np.sum((grid_points[v[0]].T * w[0]).T, axis=0 )

    close = np.argsort(np.sum((points - p)**2, axis=1))[:nclose]
    ord_close = np.argsort( hs[close] + 1j * vs[close] )
    lress, lresi, angu = [],[], []
    
    # for i in range(3):
    #     lin = points[close][ord_close][i*4:i*4+4]
    for i in range(2):
        lin = points[close][ord_close][i*2:i*2+2]

        if reverse: lres = linregress( lin[:,1], lin[:,0] )
        else: lres = linregress( lin[:,0], lin[:,1] )
        lress.append(lres[0])
        lresi.append(lres[1])
        angu.append( np.arctan(lres[0]) )

    ang = np.mean(angu) * sign + rot

    txgr, tygr = rotate_point(ang, grid_points[:,1] - pgr[1], grid_points[:,0] - pgr[0])
    return txgr, tygr

#%%
t1 = time()
vid = imageio.get_reader('Documents/Dodecahedro/Calibration/HERO9 BLACK/GX010183.MP4', 'ffmpeg') # 5000 last frame
t2 = time()

print(t2-t1)
#%%
Image_points = np.empty((0,2))
Real_pos = np.empty((0,2))

# Image 200
fima = vid.get_data(200)[:,:,1]
ny,nx = np.shape(fima)

centsx_vid, centsy_vid, red, dots = recog_dots( fima, 0, 0, hole_size=50, dot_size=3, min_size=40000, object_size=10000, max_ar=60, min_ar=5 )

vline, hline, cenh, cenv = hv_lines(dots, centsx_vid, centsy_vid, red, 0, 0, h_difference=False, v_difference=False)
filtro = (hline > 0) * (hline < 104) * (vline > -1) * (vline < 55)
order = np.argsort( -hline[filtro] + 1j * vline[filtro] )

spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]
vs, hs = vline[filtro][order] - cenv, hline[filtro][order] - cenh

txgr, tygr = real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, vs, hs, nclose=4, sign=-1, rot = 0 )

Image_points = np.vstack( (Image_points, np.vstack((spy_vid, spx_vid)).T[:]) )
Real_pos = np.vstack( (Real_pos, np.vstack((tygr, txgr)).T[:]) )
print(np.shape(Image_points))

# Image 600
fima = vid.get_data(600)[:,:,1]
centsx_vid, centsy_vid, red, dots = recog_dots( fima, 0, 0, hole_size=50, dot_size=3, min_size=40000, object_size=10000, max_ar=60, min_ar=15 )

vline, hline, cenh, cenv = hv_lines(dots, centsx_vid, centsy_vid, red, 0, 0, h_difference=False, v_difference=False)
filtro = (hline > 1) * (hline < 104) * (vline > -1) * (vline < 54)
order = np.argsort( -hline[filtro] + 1j * vline[filtro] )

spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]
vs, hs = vline[filtro][order] - cenv, hline[filtro][order] - cenh

txgr, tygr = real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, vs, hs, nclose=4, sign=-1, rot = 0 )

Image_points = np.vstack( (Image_points, np.vstack((spy_vid, spx_vid)).T[:]) )
Real_pos = np.vstack( (Real_pos, np.vstack((tygr, txgr)).T[:]) )
print(np.shape(Image_points))

# Image 2300
fima = vid.get_data(2300)[:,:,1]
centsx_vid, centsy_vid, red, dots = recog_dots( fima, 0, 0, hole_size=50, dot_size=3, min_size=40000, object_size=20000, max_ar=60, min_ar=15 )

vline, hline, cenh, cenv = hv_lines(dots, centsx_vid, centsy_vid, red, 0, 0, h_difference=False, v_difference=False)

filtro = (vline > 0) #* (hline < 104) * (vline > -1) * (vline < 55)
order = np.argsort( -hline[filtro] + 1j * vline[filtro] )

spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]
vs, hs = cenh - hline[filtro][order], vline[filtro][order] - cenv
vcopy = np.copy(vs)
vs[vcopy==-21], vs[vcopy==-20], vs[vcopy==-19] = -19, -21, -20 

txgr, tygr = real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, vs, hs, nclose=4, sign=-1, rot = 0 )

Image_points = np.vstack( (Image_points, np.vstack((spy_vid, spx_vid)).T[:]) )
Real_pos = np.vstack( (Real_pos, np.vstack((tygr, txgr)).T[:]) )
print(np.shape(Image_points))

# Image 20
fima = vid.get_data(20)[:,:,1]
centsx_vid, centsy_vid, red, dots = recog_dots( fima, 0, 0, hole_size=50, dot_size=3, min_size=40000, object_size=20000, max_ar=60, min_ar=15 )

vline, hline, cenh, cenv = hv_lines(dots, centsx_vid, centsy_vid, red, 0, 0, h_difference=False, v_difference=False, slope_ver=0.9, slope_hor=-0.9)

filtro = (vline > 1) * (hline > 0) * (hline < 54) #* (vline > -1) * (vline < 55)
order = np.argsort( -hline[filtro] + 1j * vline[filtro] )

spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]
vs, hs = hline[filtro][order] - cenh , cenv - vline[filtro][order]
vcopy, hcopy = np.copy(vs), np.copy(hs)
vs[vcopy==11] = 22
vs[ (vcopy>11)*(vcopy<=22) ] = vcopy[ (vcopy>11)*(vcopy<=22) ] - 1
vs[vcopy==24], vs[vcopy==25] = 25, 24
hs[hcopy==-25], hs[hcopy==-24] = -24, -25

txgr, tygr = real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, vs, hs, nclose=4, sign=-1, rot = 0 )

Image_points = np.vstack( (Image_points, np.vstack((spy_vid, spx_vid)).T[:]) )
Real_pos = np.vstack( (Real_pos, np.vstack((tygr, txgr)).T[:]) )
print(np.shape(Image_points))

#%%

fig, ax = plt.subplots()
ax.imshow( fima, cmap='gray' )
# ax.plot( Image_points[:5592,1], Image_points[:5592,0], '.', markersize=1)
# ax.plot( Image_points[5592:10988,1], Image_points[5592:10988,0], '.', markersize=1)
# ax.plot( Image_points[10988:16745,1], Image_points[10988:16745,0], '.', markersize=1)
# ax.plot( Image_points[16745:,1], Image_points[16745:,0], '.', markersize=1)

ax.set_axis_off()

# plt.savefig('./Documents/calibration_gopro_grid.png',dpi=400, bbox_inches='tight', transparent=False)
plt.show()

# plt.figure()
# plt.plot( Real_pos[:,1], Real_pos[:,0], '.')
# # plt.plot( Real_pos[5343:5343+5771,1], Real_pos[5343:5343+5771,0], '.')
# # plt.plot( Real_pos[5343+5771:5343+5771+5140,1], Real_pos[5343+5771:5343+5771+5140,0], '.')
# # plt.plot( Real_pos[5343+5771+5140:5343+5771+5140+5664,1], Real_pos[5343+5771+5140:5343+5771+5140+5664,0], '.')
# plt.axis('equal')
# plt.gca().invert_yaxis()
# plt.show()

tri = Delaunay(Image_points)
plt.figure()
# plt.imshow( fima ) 

plt.triplot(Image_points[:,1], Image_points[:,0], tri.simplices)
plt.plot( Image_points[:,1], Image_points[:,0], '.')

# plt.triplot(Real_pos[:,1], Real_pos[:,0], tri.simplices)
# plt.plot( Real_pos[:,1], Real_pos[:,0], '.')

plt.axis('equal')
plt.gca().invert_yaxis()
plt.show()


#%%
t1 = time()

fima = vid.get_data(20)[:,:,1]
# image = vid.get_data(4200)[40:,680:1430,2]
ny,nx = np.shape(vid.get_data(200)[:,:,1])

# plt.figure()
# plt.imshow( fima )
# plt.show()

image = fima 
xdisp, ydisp = 0, 0
hole_size= 50
dot_size= 3
min_size= 40000 
object_size= 20000

imb = image < sfr.mean(image, np.ones((15,15)) )
imo = remove_small_objects(imb, min_size=min_size)
imh = remove_small_holes(imo, area_threshold=hole_size)
imk = remove_small_objects( binary_opening(imh, disk(3)), object_size )
iml = binary_closing( np.pad( remove_small_objects(binary_opening(imk, disk(5)), 10000), 10), disk(10) )[10:-10,10:-10]

# plt.figure()
# plt.imshow( iml )
# plt.show()

mask = binary_erosion(iml, np.ones((10,10)))
imt = image * mask

ddd = sfr.mean(imt, np.ones((20,20)), mask=mask)
eee = imt - ddd*1.
normee = eee / np.max(np.abs(eee))
pokk = sfr.mean(normee, disk(1) ) #, mask=mask)
# dots = remove_small_objects(normee > threshold_otsu(normee), dot_size )
# dots = remove_small_objects(normee > threshold_yen(normee), dot_size )
dots = (gaussian(pokk,1) - gaussian(pokk,10)) > 0

# plt.figure()
# plt.imshow( dots )
# plt.show()

lal = label(dots, connectivity=1)  
props = regionprops(lal)

centsy_vid = np.array( [(props[i].centroid)[0] for i in range(len(props)) if (props[i].area<60 and props[i].area>15 ) ] ) + ydisp 
centsx_vid = np.array( [(props[i].centroid)[1] for i in range(len(props)) if (props[i].area<60 and props[i].area>15 ) ] ) + xdisp

def cent_dist(val):
    di0 = (val[0] - centsx_vid)**2  + (val[1] - centsy_vid)**2
    idi0 = np.argsort( di0 )
    return np.sum(di0[idi0[:8]] ) / di0[idi0[0]]

bounds = ( (int(np.mean(centsx_vid)-100), int(np.mean(centsx_vid)+100)), (int(np.mean(centsy_vid)-100), int(np.mean(centsy_vid)+100)) )
redi = differential_evolution(cent_dist, bounds, integrality = [True,True])

x0 = redi.x
red = least_squares(cent_dist, x0, bounds=((x0[0]-10,x0[1]-10),(x0[0]+10,x0[1]+10)) )

centsx_vid, centsy_vid, red, dots = recog_dots( image, 0, 0, hole_size=50, dot_size=3, min_size=40000, object_size=20000, max_ar=60, min_ar=15 )

vline, hline, cenh, cenv = hv_lines(dots, centsx_vid, centsy_vid, red, 0, 0, h_difference=False, v_difference=False, slope_ver=0.9, slope_hor=-0.9)

filtro = (vline > 1) * (hline > 0) * (hline < 54) #* (vline > -1) * (vline < 55)
order = np.argsort( -hline[filtro] + 1j * vline[filtro] )

spx_vid, spy_vid = (centsx_vid[filtro])[order], (centsy_vid[filtro])[order]
vs, hs = hline[filtro][order] - cenh , cenv - vline[filtro][order]
vcopy, hcopy = np.copy(vs), np.copy(hs)
vs[vcopy==11] = 22
vs[ (vcopy>11)*(vcopy<=22) ] = vcopy[ (vcopy>11)*(vcopy<=22) ] - 1
vs[vcopy==24], vs[vcopy==25] = 25, 24
hs[hcopy==-25], hs[hcopy==-24] = -24, -25


# txgr, tygr = real_grid_val(xgr1,ygr1, xgr2,ygr2, cgrx1,cgry1, cgrx2,cgry2, spx_vid,spy_vid, nclose=4, sign=-1, rot = 0 )

# Image_points = np.vstack( (Image_points, np.vstack((spy_vid, spx_vid)).T[:]) )
# Real_pos = np.vstack( (Real_pos, np.vstack((tygr, txgr)).T[:]) )

# plt.figure()
# plt.imshow( dots )
# plt.plot(centsx_vid, centsy_vid,'r.')
# plt.plot( red.x[0], red.x[1], 'g.' )
# plt.show()

t2 = time()
print(t2-t1)
#%%
ixm, iym = [],[]
xme, yme = [], []

plt.figure()
plt.imshow( dots )
for i in range(-51,53):
# # for i in range(-51,-40):
    plt.plot( spx_vid[hs==i], spy_vid[hs==i], '-' )
    xme.append( np.mean( spx_vid[hs==i]) )
    ixm.append(i)

plt.show()

plt.figure()
plt.imshow( dots )
for i in range(-27,28):
# for i in range(10,11):
    plt.plot( spx_vid[vs==i], spy_vid[vs==i], '-' )
    yme.append( np.mean(spx_vid[vs==i]) )
    iym.append(i)

plt.show()

# for j in range(0,28):
#     plt.figure()
#     plt.imshow( dots )
#     plt.title(j)
#     for i in range(-3,j):
#         plt.plot( spx_vid[vs==i], spy_vid[vs==i], '-' )
#         # plt.plot( spx_vid[hs==i], spy_vid[hs==i], '-' )
#     plt.show()


# plt.figure()
# plt.plot(ixm,xme,'.-')
# plt.plot(iym,yme,'.-')
# plt.grid()
# plt.show()


#%%



hline = np.zeros_like( centsx_vid ) * np.nan
for i in tqdm(range(len(list_hor_lines0))):
    u = np.linspace(0,1500,3000)
    (a,b,c,d),cov = curve_fit(pol3, list_hor_lines0[i][:,1], list_hor_lines0[i][:,0])    
    
    plt.plot( list_hor_lines0[i][:,1] + xdisp, list_hor_lines0[i][:,0] + ydisp, '-' )
plt.show()
    
plt.figure()
plt.imshow( fima )

vline = np.zeros_like( centsx_vid ) * np.nan
for i in tqdm(range(len(list_ver_lines0))):
    u = np.linspace(0,1500,3000) 
    (a,b,c,d),cov = curve_fit(pol3, list_ver_lines0[i][:,0], list_ver_lines0[i][:,1])

    plt.plot( list_ver_lines0[i][:,1] + xdisp, list_ver_lines0[i][:,0] + ydisp, '-' )
plt.show()    
    








#%%












#%%










