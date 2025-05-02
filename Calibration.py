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

fin = None

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


fin = 10

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









#%%



#%%









#%%