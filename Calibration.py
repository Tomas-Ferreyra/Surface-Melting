#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:48:36 2025

@author: tomasferreyrahauchar
"""

import numpy as np
import matplotlib.pyplot as plt

import imageio.v2 as imageio
# import imageio.v3 as iio
from tqdm import tqdm
from time import time

from scipy.optimize import least_squares, differential_evolution, curve_fit

from skimage.morphology import remove_small_objects, binary_dilation, disk, skeletonize, binary_closing, remove_small_holes, binary_opening, binary_erosion
from skimage.measure import label, regionprops
from skimage.filters import try_all_threshold, threshold_otsu, gaussian
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
xgr, ygr = np.arange(55), np.arange(82)
xgr,ygr = np.meshgrid(xgr,ygr)
cgrx, cgry = xgr.flatten(), ygr.flatten()
cgrx2, cgry2 = cgrx - cgrx[din2+1], cgry - cgry[din2+1]

t2 = time()
t2-t1
#%%

fin = None

plt.figure()
plt.imshow(gridf1, cmap='gray')
plt.plot(spx1[:fin], spy1[:fin], 'r.')
plt.plot(cx1, cy1, 'g.')
plt.show()

plt.figure()
plt.imshow(gridf2, cmap='gray')
plt.plot(spx2[:fin], spy2[:fin], 'r.')
plt.plot(cx2, cy2, 'g.')
plt.show()


plt.figure()
plt.plot(spx1[:fin] - cx1, spy1[:fin] - cy1,'r.')
plt.plot(cx1-cx1, cy1-cy1,'b.')
plt.grid()
plt.gca().invert_yaxis()
plt.show()
plt.figure()
plt.plot(spx2[:fin] - cx2, spy2[:fin] - cy2,'r.')
plt.plot(cx2-cx2, cy2-cy2,'b.')
plt.grid()
plt.gca().invert_yaxis()
plt.show()

plt.figure()
plt.plot(cgrx1[:fin], cgry1[:fin], '.')
plt.plot(cgrx1[din1+1], cgry1[din1+1], '.')
plt.axis('equal')
plt.grid()
plt.gca().invert_yaxis()
plt.show()
plt.figure()
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

diference_grids(cgrx1,cgry1,cgrx2,cgry2, spx1-cx1,spy1-cy1, spx2-cx2, spy2-cy2)

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

centsy = np.array( [(props[i].centroid)[0] for i in range(len(props)) ] ) + 180
centsx = np.array( [(props[i].centroid)[1] for i in range(len(props)) ] ) + 275


bounds = ( (int(np.mean(rcx)-100), int(np.mean(rcx)+100)), (int(np.mean(rcy)-100), int(np.mean(rcy)+100)) )
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
hline = np.zeros_like( centsx ) * np.nan

for i in tqdm(range(len(list_hor_lines0))):
    u = np.linspace(0,1500,3000)
    (a,b,c,d),cov = curve_fit(pol3, list_hor_lines0[i][:,1], list_hor_lines0[i][:,0])    
    
    dis = []
    for n in range(len(centsx)):
        point = [centsx[n] - 275, centsy[n] - 180]
        poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )
        dis.append( poop )
    dis = np.array(dis)
    
    fill = dis < 20
    sorting = np.argsort( centsx[fill] )
    orhgx.append( (centsx[fill])[sorting] )
    orhgy.append( (centsy[fill])[sorting] )
    hline[fill] = i

orvgx, orvgy = [], []
vline = np.zeros_like( centsx ) * np.nan

for i in tqdm(range(len(list_ver_lines0))):
    u = np.linspace(0,1500,3000) 
    (a,b,c,d),cov = curve_fit(pol3, list_ver_lines0[i][:,0], list_ver_lines0[i][:,1])    
    
    dis = []
    for n in range(len(centsy)):
        point = [centsx[n] - 275, centsy[n] - 180]
        poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d) , u, point)  )
        dis.append( poop )
    dis = np.array(dis)
    
    fill = dis < 20
    sorting = np.argsort( centsy[fill] )
    orvgx.append( (centsx[fill])[sorting] )
    orvgy.append( (centsy[fill])[sorting] )
    vline[fill] = i

filtro = vline > 0
order = np.argsort( vline[filtro] + 1j * hline[filtro] )

t2 = time()
print()
print(t2-t1)

#%%
spx, spy = (centsx[filtro])[order], (centsy[filtro])[order]

fin = 500

plt.figure()
# plt.imshow(fima, cmap='gray')

# plt.plot(nx/2,ny/2,'g.' )
plt.plot( centsx, centsy, 'r.', markersize=10 )

plt.plot( spx[:fin], spy[:fin], 'b.', markersize=5 )

plt.show()

































#%%

# vid = iio.imread('Documents/Dodecahedro/Calibration/HERO9 BLACK/GX010175.MP4', idx=1, plugin="pyav")
# 
# import cv2

# vid = cv2.VideoCapture('Documents/Dodecahedro/Calibration/HERO9 BLACK/GX010175.MP4')

# for i in range(30):
#     ss,image = vid.read()
#     print(ss, end=' ')

t1 = time()
vid = imageio.get_reader('Documents/Dodecahedro/Calibration/HERO9 BLACK/GX010175.MP4', 'ffmpeg') # 5000 last frame

# imed = []
# for i in tqdm(range(5001)):
#     image = vid.get_data(i)
#     imed.append( image[180:1010,275:1710,2] )

# imed = np.median(imed, axis=0)
# imed = np.mean(imed, axis=0)
t2 = time()
print(t2-t1)
#%%
fima = vid.get_data(1000)[:,:,2]
image = vid.get_data(1000)[180:1010,275:1710,2]
ny,nx = np.shape(vid.get_data(1000)[:,:,2])

plt.figure()
plt.imshow(image)
plt.show()
#%%
t1 = time()
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

# ecc = [props[i].axis_major_length for i in range(len(props)) ]
centsy = np.array( [(props[i].centroid)[0] for i in range(len(props)) ] ) + 180
centsx = np.array( [(props[i].centroid)[1] for i in range(len(props)) ] ) + 275

# sor_points = order_points(centsx, centsy, max_iter=200, distance=5) # en px
# spx, spy = sor_points[:,0], sor_points[:,1]
t2 = time()
t2-t1
#%%
# center of calibration grid (where there is no dot)
theta = 0 #-70
rcx, rcy = rotate_point(theta * np.pi/180, centsx, centsy)


t2 = time()
print(t2-t1)


#%%

plt.figure()
# plt.imshow(dots)
# plt.imshow(fima, cmap='gray')

# plt.plot(nx/2,ny/2,'g.' )
plt.plot( centsx, centsy, 'r.' )
plt.show()

#%%
theta = np.pi/180 * 70
scale = 11

grx = np.array( [ i for j in range(-53,54) for i in range(-29,30) ])
gry = np.array( [ j for j in range(-53,54) for i in range(-29,30) ])

rgx, rgy = rotate_point(theta, grx, gry)
rgx, rgy = rgx * scale + red.x[0], rgy * scale + red.x[1]

rgr, rgt = np.sqrt( (rgx - nx/2)**2 + (rgy - ny/2)**2), np.arctan2(rgy-ny/2, rgx-nx/2)
k = [0, 1, -0.0, 0.01]
rgr = k[0] + k[1] * rgr + k[2] * rgr**2 + k[3] * rgr**3


por, pot = np.sqrt( (centsx - nx/2)**2 + (centsy -ny/2)**2 ), np.arctan2( centsy-ny/2, centsx-nx/2)

plt.figure()
# plt.plot( rgx - nx/2, rgy - ny/2, '.')
# plt.plot( rgr * np.cos(rgt), rgr * np.sin(rgt), '.', markersize=2)

plt.plot(centsx - nx/2, centsy -ny/2, '.' )
plt.plot( por * np.cos(pot), por * np.sin(pot), '.', markersize=2 )

plt.show()
#%%

distc = dist( centsx, centsy, red.x )
idi = np.argsort(distc)

dcx, dcy = centsx[idi[:4]], centsy[idi[:4]]
distc1 = dist( dcx, dcy, [dcx[0],dcy[0]] )
idc = np.argmax(distc1)

pinx, piny = list(dcx[[0,idc]]), list(dcy[[0,idc]])
pinx.append( red.x[0] )
piny.append( red.x[1] )


(a,b,c),cov = curve_fit(pol2, pinx, piny)

u = np.linspace(900,1100, 1000)
    
plt.figure()
plt.plot( centsx, centsy, 'r.', markersize=10)
plt.plot( centsx[idi[:4]], centsy[idi[:4]], 'g.', markersize=5 )
plt.plot( pinx, piny, 'm.', markersize=3 )
plt.plot( u, pol2(u, a, b, c) )
# plt.plot( np.sort(disf),'.-')
plt.show()


#%%

centsx




#%%
# def dists2_cuad_point(a,b,c,x, point ):
#     return (x - point[0])**2 + (a+b*x+c*x**2 - point[1])**2

# def xroot(a,b,c,point):
#     px,py = point[0], point[1]
#     c1 = 2*a*b - 2*px - 2*b*py
#     c2 = 2 + 2*b**2 + 4*a*c - 4*c*py
#     c3 = 6*b*c 
#     c4 = 4*c**2

#     coff = [c4,c3,c2,c1] #[c1,c2,c3,c4]    

#     root = np.roots(coff)
#     indi = np.argmin( dists2_cuad_point(a, b, c, root, point) )
#     xro = np.real(root[indi])
#     return xro

# def point_to_caud(a,b,c, point):
#     xro = xroot(a, b, c, point)
#     return dists2_cuad_point(a,b,c,xro, point )


(a,b,c),cov = curve_fit(pol2, pinx, piny)

dists = [] 
for i in range(len(centsx)):
    dista = point_to_caud(a, b, c, [centsx[i],centsy[i]])
    distap = np.min( (np.array(pinx) - centsx[i])**2 + (np.array(piny) - centsy[i])**2 )
    dists.append(dista + distap)
prox = np.argsort(dists)[2]

pinx.append(centsx[])

u = np.linspace(800,1100, 1000)

plt.figure()
plt.plot( centsx, centsy, 'r.', markersize=10)

plt.plot( pinx, piny, 'm.', markersize=3 )

plt.plot( u, pol2(u, a, b, c) )

plt.plot( centsx[prox], centsy[prox], 'g.', markersize=3 )

# plt.plot( np.sort(disf),'.-')
plt.show()

#%%


















#%%

#%%
theta = -70
rcx, rcy = rotate_point(theta * np.pi/180, centsx, centsy)

# puntos = np.hstack((np.array([centsx]).T, np.array([centsy]).T, np.array([[0]*len(centsy)]).T ))
puntos = np.hstack((np.array([rcx]).T, np.array([rcy]).T, np.array([[0]*len(rcy)]).T ))
mima = [np.argmin(puntos[:,0]+puntos[:,1]), np.argmax(puntos[:,0]-puntos[:,1])]
or_puntos = np.empty((0,3), int)

# plt.figure()
# plt.plot( centsx, centsy, 'ro')
# # plt.plot( centsx[mima], centsy[mima], 'g.-')
# plt.show()

cxa, cya = np.mean(rcx), np.mean(rcy)

# dists = (cxa - rcx)**2  + (cya - rcy)**2 
# idis = np.argsort(dists)[0]

di4 = []
for i in tqdm(range(len(rcx))):
    di0 = (rcx[i] - rcx)**2  + (rcy[i] - rcy)**2
    idi0 = np.argsort( di0 )
    # di4 += list( di0[idi0][:9] )
    di4.append( list( di0[idi0][:9] ) )

# plt.figure()

# plt.plot( rcx[idi0[:5]], rcy[idi0[:5]], 'go' )

# plt.plot( rcx, rcy, 'r.')
# plt.plot( cxa,cya, "g." )

# # plt.plot( rcx[mima], rcy[mima], 'g.-')
# # plt.axis('equal')
# plt.title( theta )
# plt.show()

#%%

theta = -70
# rcx, rcy = rotate_point(theta * np.pi/180, centsx - red.x[0], centsy - red.x[1])
rcx, rcy = rotate_point(theta * np.pi/180, centsx - nx/2, centsy - ny/2)
rcr = np.sqrt(rcx**2 + rcy**2)

puntos = np.hstack((np.array([rcx]).T, np.array([5*rcy]).T, np.array([[0]*len(rcy)]).T ))
mima = [np.argmin(puntos[:,0]+puntos[:,1]), np.argmax(puntos[:,0]-puntos[:,1])]
or_puntos = np.empty((0,3), int)



div = rcr * 0.000001

plt.figure()

plt.plot(rcx,rcy,'.')
# plt.plot(rcx * div, rcy * div,'.')

# plt.plot(rcx,5*rcy- 40,'.')
# plt.plot(rcx[mima],5*rcy[mima],'.')

# plt.axis('equal')
plt.show()



#%%
import matplotlib.tri as tri

theta = -70
rcx, rcy = rotate_point(theta * np.pi/180, centsx - red.x[0], centsy - red.x[1])

trr = tri.Triangulation(rcx, rcy)

# plt.figure()
# # plt.plot(di4, '.-')
# tri.triplot(plt.axes(), trr )
# plt.plot( rcx,rcy, 'r.' )

plt.show()

coordinates = np.column_stack((rcx,rcy))
e_start = coordinates[trr.edges[:,0]]
e_end = coordinates[trr.edges[:,1]]
e_diff = e_end - e_start
e_x = e_diff[:,0]
e_y = e_diff[:,1]

e_len = np.sqrt(e_x**2+e_y**2)
alpha = 180*np.arcsin(e_y/e_len)/np.pi

ind_horizontal = (-10<alpha) & (alpha < 10)
edges_horizontal = trr.edges[ind_horizontal]


edges_horizontal.flatten()
plt.figure()
plt.hist(alpha, bins=50)
plt.show()

fin = 50
plt.figure()
# plt.plot(centsx - red.x[0], centsy - red.x[1], '.')
plt.plot(rcx, rcy, '.')
# plt.plot( rcx[edges_horizontal[:,0]], rcy[edges_horizontal[:,1]], '.' )
plt.show()





#%%
theta = 0 #-70
rcx, rcy = rotate_point(theta * np.pi/180, centsx, centsy)

def cent_dist(val):
    di0 = (val[0] - rcx)**2  + (val[1] - rcy)**2
    idi0 = np.argsort( di0 )
    return np.sum(di0[idi0[:8]] ) / di0[idi0[0]]

t1 = time()
bounds = ( (int(np.mean(rcx)-100), int(np.mean(rcx)+100)), (int(np.mean(rcy)-100), int(np.mean(rcy)+100)) )
redi = differential_evolution(cent_dist, bounds, integrality = [True,True])

x0 = redi.x
red = least_squares(cent_dist, x0, bounds=((x0[0]-10,x0[1]-10),(x0[0]+10,x0[1]+10)) )
t2 = time()
print(t2-t1)

plt.figure()
plt.plot(rcx, rcy, 'r.', markersize=1)
plt.plot( x0[0], x0[1], 'g.' )
# plt.plot( redi.x[0], redi.x[1], 'b.' )
plt.plot( red.x[0], red.x[1], 'm.' )
plt.gca().invert_yaxis()
plt.show()

redi.x, redi.fun, red.x, red.fun
#%%
val = red.x
disf = (val[0] - rcx)**2  + (val[1] - rcy)**2

idi = np.argsort(disf)

plt.figure()
plt.plot( rcx, rcy, 'r.', markersize=10)
plt.plot( rcx[idi[:4]], rcy[idi[:4]], 'g.', markersize=5 )
# plt.plot( np.sort(disf),'.-')
plt.show()

#%%
n = -100
val = [rcx[n], rcy[n]]
dis = (val[0] - rcx)**2  + (val[1] - rcy)**2

idi = np.argsort(dis)
fin = 5

plt.figure()
plt.plot( rcx, rcy, 'r.', markersize=10)
plt.plot( val[0], val[1], 'b.', markersize=10)
plt.plot( rcx[idi[:fin]], rcy[idi[:fin]], 'g.', markersize=5 )
# plt.plot( np.sort(disf),'.-')
plt.show()
#%%
grx = np.array( [ i for j in range(-53,54) for i in range(-29,30) ])
gry = np.array( [ j for j in range(-53,54) for i in range(-29,30) ])

def gridp(val):
    rgrx = (np.cos(val[0]) * grx - np.sin(val[0]) * gry) * val[1]
    rgry = (np.sin(val[0]) * grx + np.cos(val[0]) * gry) * val[1] 
    
    # pgrx = val[2] * rgrx + val[4] * rgrx**2
    # pgry = val[3] * rgry + val[5] * rgry**2
    
    pgrx = (val[2] * rgrx + val[4] * rgry + val[6]) / (val[8] * rgrx + val[9] * rgry + 1)
    pgry = (val[3] * rgrx + val[5] * rgry + val[7]) / (val[8] * rgrx + val[9] * rgry + 1)
    
    return pgrx, pgry
#%%

grx = np.array( [ i for j in range(-53,54) for i in range(-29,30) ])
gry = np.array( [ j for j in range(-53,54) for i in range(-29,30) ])

def grid_match(val):
    rgrx = (np.cos(val[0]) * grx - np.sin(val[0]) * gry) * val[1]
    rgry = (np.sin(val[0]) * grx + np.cos(val[0]) * gry) * val[1] 
    
    # pgrx = val[2] * rgrx + val[4] * rgrx**2
    # pgry = val[3] * rgry + val[5] * rgry**2
    
    pgrx = (val[2] * rgrx + val[4] * rgry + val[6]) / (val[8] * rgrx + val[9] * rgry + 1)
    pgry = (val[3] * rgrx + val[5] * rgry + val[7]) / (val[8] * rgrx + val[9] * rgry + 1)
    
    sdis = []
    for i in range(len(rcx)):
        xpr = pgrx[i] - red.x[0]
        ypr = pgry[i] - red.x[1]
        
        drg = np.min( (rcx - xpr)**2 + (rcy - ypr)**2 )
        sdis.append( drg )
    return np.sum(drg)
    
# x0 = [1.348, 12.03, 1, 1, 1, 1,1,1,1,1,1,1 ]
x0 = [0, 1, 1, 1, 1, 1,1,1,1,1 ]
ser = least_squares(grid_match , x0)

ser
#%%

# x0 = [1.35, 12, 1,1, 1e-10,1e-3]
# x0 = [0, 12, 1,1, 1e-10,1e-3]
x0 = ser.x
ggrx, ggry = gridp(x0)

plt.figure()
plt.plot(rcx - red.x[0], rcy - red.x[1], 'r.', markersize=8)
plt.plot(ggrx, ggry, 'b.', markersize=2)
plt.show()

#%%
val = [0, 12, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1]
grx = np.array( [ i for j in range(-53,54) for i in range(-29,30) ])
gry = np.array( [ j for j in range(-53,54) for i in range(-29,30) ])

rgrx = (np.cos(val[0]) * grx - np.sin(val[0]) * gry) * val[1]
rgry = (np.sin(val[0]) * grx + np.cos(val[0]) * gry) * val[1] 

pgrx = (val[2] * rgrx + val[4] * rgry + val[6]) / (val[8] * rgrx + val[10] * rgry + 1)
pgry = (val[3] * rgrx + val[5] * rgry + val[7]) / (val[9] * rgrx + val[11] * rgry + 1)

plt.figure()
# plt.plot( grx,gry, '.' )
# plt.plot( rgrx,rgry, '.' )
plt.plot( pgry,pgry, '.' )
plt.show()
#%%

def calibration_first(v, x,y):
    xtop = v[0]*x + v[1]*y + v[2] 
    ytop = v[3]*x + v[4]*y + v[5] 
    bot = v[6]*x + v[7]*y + 1 

    X,Y = xtop / bot, ytop / bot
    return X,Y

def calibration_second(v, x,y):
    xtop = v[0]*x + v[1]*y + v[2] + v[3]*x**2 +  v[4]*y**2 +  v[5]*x*y
    ytop = v[6]*x + v[7]*y + v[8] + v[9]*x**2 + v[10]*y**2 + v[11]*x*y
    bot = v[12]*x + v[13]*y + 1 + v[14]*x**2 + v[15]*y**2 + v[16]*x*y

    X,Y = xtop / bot, ytop / bot
    return X,Y

def residual_first(v, spx,spy, grx, gry):
    rspx = (np.cos(val[8]) * spx - np.sin(val[8]) * spx) #* val[1]
    rspy = (np.sin(val[8]) * spy + np.cos(val[8]) * spy) #* val[1] 
    
    xtop = v[0]*rspx + v[1]*rspy + v[2]
    ytop = v[3]*rspx + v[4]*rspy + v[5]
    bot = v[6]*rspx + v[7]*rspy + 1 

    X,Y = xtop / bot, ytop / bot
    
    suma = 0
    for i in range(len(grx)):
        suma += (X - grx[i])**2 + (Y - gry[i])**2
    return suma

def residual_second(v, spx,spy, grx, gry):
    xtop = v[0]*spx + v[1]*spy + v[2] + v[3]*spx**2 +  v[4]*spy**2 +  v[5]*spx*spy
    ytop = v[6]*spx + v[7]*spy + v[8] + v[9]*spx**2 + v[10]*spy**2 + v[11]*spx*spy
    bot = v[12]*spx + v[13]*spy + 1 + v[14]*spx**2 + v[15]*spy**2 + v[16]*spx*spy

    X,Y = xtop / bot, ytop / bot
    return (X - grx)**2 + (Y - gry)**2
#%%
t1 = time()
res_first = lambda v: residual_first(v, grx, gry, (rcx - red.x[0]), (rcy - red.x[1]) )

ls1 = least_squares(res_first , [1,1,1,1,1,1,0,0,1.35], method='lm')

t2 = time()
print(t2-t1)


# ls1 = least_squares(residual_first , [1,1,1,1,1,1,0,0], method='lm')

# ls0 = [0.]*17
# ls0[:3], ls0[6:9], ls0[12:14] = ls1.x[:3], ls1.x[3:6], ls1.x[6:]

# ls2 = least_squares(residual_second, ls0, method='lm')

# t2 = time()
# t2-t1, ls1, ls2
#%%

cgrx, cgry = calibration_first(ls1.x, grx,gry) 

plt.figure()
plt.plot( (rcx - red.x[0]), (rcy - red.x[1]), 'r.')
plt.plot(cgrx, cgry, 'b.')
plt.show()


#%%


plt.figure()
plt.imshow( vid.get_data(1000)[:,:,2], cmap='gray')
plt.plot( nx/2, ny/2, '.' )
plt.show()

#%%

theta = 0 #-70
rcx, rcy = rotate_point(theta * np.pi/180, centsx - nx/2, centsy - ny/2)

grx = np.array( [ i for j in range(-53,54) for i in range(-29,30) ])
gry = np.array( [ j for j in range(-53,54) for i in range(-29,30) ])

ang = 75
rgx, rgy = rotate_point(ang * np.pi/180, grx, gry)
grr = np.sqrt(rgx**2 + rgy**2)

ks = [0.1, 0.0001]
polk = 0
for i in range(len(ks)): 
    polk += ks[i] * grr**i

drx = (rgx + 0*(red.x[0]-nx/2) ) / polk
dry = (rgy + 0*(red.x[1]-ny/2) ) / polk

plt.figure()
# plt.plot( rgx, rgy, '.')
plt.plot( drx, dry, '.')

plt.plot( rcx, rcy, '.')

# plt.plot( rcx, rcy, '.')

# plt.axes('equal')
plt.show()



#%%
theta = 0 #-70
rcx, rcy = rotate_point(theta * np.pi/180, centsx - nx/2, centsy - ny/2)

grx = np.array( [ i for j in range(-53,54) for i in range(-29,30) ])
gry = np.array( [ j for j in range(-53,54) for i in range(-29,30) ])


def residues(val):
    rgx, rgy = rotate_point(val[0] * np.pi/180, grx, gry)
    grr = np.sqrt(rgx**2 + rgy**2)

    ks = [val[1], val[2]]
    polk = 0
    for i in range(len(ks)): 
        polk += ks[i] * grr**i
    
    # drx = (rgx + val[3] ) / polk
    # dry = (rgy + val[4]) / polk
    drx = (rgx ) / polk
    dry = (rgy ) / polk
   
    suma = 0
    for i in range(len(rcx)):
        did = np.median( (rcx[i]-drx)**2 + (rcy[i]-dry)**2 )
        suma += did
    return suma


# reis = least_squares(residues, [75,0.08,0.001, red.x[0]-nx/2, red.x[1]-ny/2])
reis = least_squares(residues, [75,0.08,0.0001]) #, bounds=((50,1e-4),(90,2)))


reis
#%%
val = reis.x
rgx, rgy = rotate_point(val[0] * np.pi/180, grx, gry)
grr = np.sqrt(rgx**2 + rgy**2)

ks = [val[1], val[2]]
# ks = [val[1], 0.0005]
polk = 0
for i in range(len(ks)): 
    polk += ks[i] * grr**i

# drx = (rgx + val[3] ) / polk
# dry = (rgy + val[4] ) / polk
drx = (rgx ) / polk
dry = (rgy ) / polk

plt.figure()
plt.plot( drx, dry, '.')
plt.plot( rcx, rcy, '.')
# plt.axes('equal')
plt.show()



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

orhgx, orhgy = [], []
hline = np.zeros_like( centsx ) * np.nan

for i in tqdm(range(len(list_hor_lines0))):
    u = np.linspace(0,1500,3000)
    (a,b,c,d),cov = curve_fit(pol3, list_hor_lines0[i][:,1], list_hor_lines0[i][:,0])    
    
    dis = []
    for n in range(len(centsx)):
        point = [centsx[n] - 275, centsy[n] - 180]
        poop = np.min( distspol_cuad_point(u, pol3(u, a, b, c, d), point)  )
        dis.append( poop )
    dis = np.array(dis)
    
    fill = dis < 20
    sorting = np.argsort( centsx[fill] )
    orhgx.append( (centsx[fill])[sorting] )
    orhgy.append( (centsy[fill])[sorting] )
    hline[fill] = i

orvgx, orvgy = [], []
vline = np.zeros_like( centsx ) * np.nan

for i in tqdm(range(len(list_ver_lines0))):
    u = np.linspace(0,1500,3000) 
    (a,b,c,d),cov = curve_fit(pol3, list_ver_lines0[i][:,0], list_ver_lines0[i][:,1])    
    
    dis = []
    for n in range(len(centsy)):
        point = [centsx[n] - 275, centsy[n] - 180]
        poop = np.min( distspol_cuad_point(pol3(u, a, b, c, d) , u, point)  )
        dis.append( poop )
    dis = np.array(dis)
    
    fill = dis < 20
    sorting = np.argsort( centsy[fill] )
    orvgx.append( (centsx[fill])[sorting] )
    orvgy.append( (centsy[fill])[sorting] )
    vline[fill] = i

order = np.argsort( vline + 1j * hline )

t2 = time()
print()
print(t2-t1)
#%%

hline[-5:]
valline[-5:]










