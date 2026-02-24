#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 11:56:09 2026

@author: tomasferreyrahauchar
"""

# =============================================================================
# Energies comparison
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq #fsolve
from scipy.integrate import cumulative_trapezoid
# from scipy.signal import convolve, fftconvolve
# from time import time


def exp_convolution(t, T, beta):
    """
    Compute y(t) = ∫_0^t exp(-beta (t-τ)) T(τ) dτ

    Parameters
    ----------
    t : 1D numpy array
        Time array (must be uniformly spaced)
    T : 1D numpy array
        Signal values at times t
    beta : float
        Positive decay constant

    Returns
    -------
    y : 1D numpy array
        Convolution result
    """
    # beta = alpha * lam**2
    t = np.asarray(t)
    T = np.asarray(T)

    dt = t[1] - t[0]
    y = np.zeros_like(T)

    # Exact discrete update for exponential kernel
    decay = np.exp(-beta * dt)
    coeff = (1 - decay) / beta

    for n in range(len(t) - 1):
        y[n+1] = decay * y[n] + coeff * T[n]

    return y


def find_roots(L, h, n, tol=1e-10):
    """
    Compute the first n positive roots of:
        tan(L*x) = 1/(h*x)

    Parameters
    ----------
    L : float
        Parameter L in tan(L*x)
    h : float
        Parameter h in 1/(h*x)
    n : int
        Number of positive roots to compute
    tol : float
        Tolerance for root-finding

    Returns
    -------
    roots : list of floats
        First n positive roots
    """
    
    def f(x):
        # return np.tan(L * x) * h  - 1. / x
        return np.sin(L * x) * h * x - np.cos(L * x)

    roots = []
    k = 0
    
    while len(roots) < n:
        # Asymptotes of tan(Lx)
        # left = ((k+0.5 - 0.5) * np.pi) / L
        # right = ((k + 1.5 - 1) * np.pi) / L
        left = ((k) * np.pi) / L
        right = ((k + 1.) * np.pi) / L
        
        # Avoid singularities
        a = left + 1e-12
        b = right - 1e-12
        
        try:
            if f(a) * f(b) < 0:
                root = brentq(f, a, b, xtol=tol)
                if root > 0:
                    roots.append(root)
        except ValueError:
            pass
        
        k += 1

    return np.array(roots)


def sn(lam,h,L):
    bot = lam * (  L + h + h**2 * L * lam**2 )
    return 2 / bot 


def bt_nm(lam,t,T,alpha,h,L):
    s = sn(lam,h,L)
    beta = alpha * lam**2
    ct = exp_convolution(t, T, beta)
    t1 = np.exp(-beta*t) 
    return s/(lam*L) * ( t1 - T + beta * ct )

def bt_n(lam,t,T,alpha,h,L):
    s = sn(lam,h,L)
    beta = alpha * lam**2
    ct = exp_convolution(t, T, beta)
    t1 = T[0] * np.exp(-beta*t) 
    return s * ( t1 - T + beta * ct )


def sol_heateq(x,t,T,N,alpha,h,L):
    if h == 0:
        lam = np.pi/2/L * (1 + 2* np.arange(0,N)) 
        ttot = 0
        for n in range(0,N):
            b = np.expand_dims( bt_n(lam[n], t, T, alpha,0,L), axis=1)
            sino = np.sin(lam[n] * x)
            # cose = np.cos(lam[n] * x)
    
            ttot += b * sino 
        return ttot

    else:
        lam = find_roots(L, h, N)
        ttot = 0
        for n in range(0,N):
            b = np.expand_dims( bt_n(lam[n], t, T, alpha,h,L), axis=1)
            sino = np.sin(lam[n] * x)
            cose = np.cos(lam[n] * x)
    
            ttot += b * ( h*lam[n]*cose + sino )
        return ttot



def energy_delayed(t,T, N, alpha,h,L):

    lam = find_roots(L, h, N)
    mb = np.zeros_like(t)
    for n in range(0,N):        
        mbn = bt_nm(lam[n], t, T, alpha,h,L) 
        
        mb += mbn
        
    # print( mb[1:] )
    return 1 - T - mb

def energy_loss(t,T):
    return cumulative_trapezoid((1-T),t, initial=0)
    
def energy_bath(t,T):
    return 1-T

#%%
filepath = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/E_balance/'
save = 0

n_t = 100
c = 60**2 / 10
t = np.linspace(0, 60**2*2, n_t)
T = np.exp(- t / c) * (1-6/20)+6/20

name = 'temperature profile (adim)'
plt.figure()
plt.plot( t/60 , T)
plt.xlabel('t (min)')
plt.ylabel('T(t) (°C)')
if save: plt.savefig(filepath+name+'.pdf', dpi=200, bbox_inches='tight')
plt.show()

cpw, cpai, cpac, cpi = 4184, 1005, 1466, 2110 # J/(kg K)
Hd = 34320 # J/K, total heat capacity of dodecahedron
rhow, rhoi, rhoai, rhoac = 998.2, 916.8, 1.225, 1180  # kg/m3
latent = 334000 # J/kg 

Vb, V0 = 112/rhow, 20/rhoi
k_loss = 12.58 * (1 + Hd/(cpw*112) ) #J/(s K)


diff_ai, diff_ac = 19 / 1e6, 0.12 / 1e6 # mm^2/s (/1e6 to make it m^2/), diffusivity air and acrylic
k_ai, k_ac = 0.024, 0.19 # W/(m K), conductivity

h_ai, h_ac = 10, 300 # W/(m^2 K)
# h_ai, h_ac = 1e10, 1e10 # W/(m^2 K)

L1_ai, L2_ai, L_ac = 0.03, 0.3, 0.16 # m, thickness of air layer or acrylic
Ar_ai = 0.36 # m^2
m_ac = 8 * 2.6 # kg
Tice = -1 #°C , initial ice temp 

#Without prefactors 
Ebath = energy_bath(t, T)
Edodec = energy_bath(t, T)
Eloss = energy_loss(t, T) / 5000

Eair_lid =  energy_delayed(t, T, 5000, diff_ai, k_ai/h_ai, L1_ai )
Eair_nlid = energy_delayed(t, T, 5000, diff_ai, k_ai/h_ai, L2_ai )
Eac = energy_delayed(t, T, 5000, diff_ac, k_ac/h_ac, L_ac )

name = 'Energies (only time dependance)'
plt.figure()
plt.plot(t, Ebath, label=r'$E_{bath}$')
plt.plot(t, Edodec, label=r'$E_{dodec}$')
plt.plot(t, Eloss, label=r'$E_{loss} \, / \, 5000$')
plt.plot(t, Eair_lid, label=r'$E_{air}$ (with lid)')
plt.plot(t, Eair_nlid, label=r'$E_{air}$ (without lid)')
plt.plot(t, Eac, label=r'$E_{acrylic}$')
plt.xlabel(r'$t$ (seg)')
plt.ylabel(r'$E/(\rho_w \, V_b \, c_{p,w} ))$ (seg)')
plt.legend()
if save: plt.savefig(filepath+name+'.pdf', dpi=200, bbox_inches='tight')
plt.show()

#With prefactors 
Ebath = energy_bath(t, T)
Edodec = Hd/(rhow*Vb*cpw) * energy_bath(t, T)
Eloss = k_loss/(rhow*Vb*cpw) * energy_loss(t, T)

Eair_lid = (rhoai*Ar_ai*L1_ai*cpai)/(rhow*Vb*cpw) * energy_delayed(t, T, 5000, diff_ai, k_ai/h_ai, L1_ai )
Eair_nlid = (rhoai*Ar_ai*L2_ai*cpai)/(rhow*Vb*cpw) * energy_delayed(t, T, 5000, diff_ai, k_ai/h_ai, L2_ai )
Eac = (m_ac*cpac)/(rhow*Vb*cpw) * energy_delayed(t, T, 5000, diff_ac, k_ac/h_ac, L_ac )

name = 'Adimensional energies'
plt.figure()
plt.plot(t, Ebath, label=r'$E_{bath}$')
plt.plot(t, Edodec, label=r'$E_{dodec}$')
plt.plot(t, Eloss, label=r'$E_{loss}$')
plt.plot(t, Eair_lid, label=r'$E_{air}$ (with lid)')
plt.plot(t, Eair_nlid, label=r'$E_{air}$ (without lid)')
plt.plot(t, Eac, label=r'$E_{acrylic}$')
plt.xlabel(r'$t$ (seg)')
plt.ylabel(r'$E/(\rho_w \, V_b \, c_{p,w} ))$ (seg)')
plt.legend()
if save: plt.savefig(filepath+name+'.pdf', dpi=200, bbox_inches='tight')
plt.show()

Elat = latent / (cpw*20) * np.ones_like(t)
Em = T * np.ones_like(t)
Eice = -cpi/cpw * Tice/20 * np.ones_like(t)

print( np.max(Eice/Elat), np.min(Em/Elat) )
# Energy to heat ice is 3% (with T_i=-5°C) or 0.6% (with T_i=-1°C) of latent heat
# Energy to heat up melted water is at least 7.5% of latent

name = 'Ice volume related energies'
plt.figure()
plt.plot(t, Elat, label=r'$E_{latent}$')
plt.plot(t, Em, label=r'$E_{melted}$')
plt.plot(t, Eice, label=r'$E_{ice}$')
plt.xlabel(r'$t$ (seg)')
plt.ylabel(r'$E \, \gamma/(\beta (1-\tilde{V}))$ (seg)')
plt.legend()
if save: plt.savefig(filepath+name+'.pdf', dpi=200, bbox_inches='tight')
plt.show()



#%%
# Sols heat eq
save = 0
filepath = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/E_balance/'

h_ai, h_ac = 10, 300 # W/(m^2 K)
# h_ai, h_ac = 1e16, 1e16 # W/(m^2 K)

diff_ai, diff_ac = 19 / 1e6, 0.12 / 1e6 # mm^2/s (/1e6 to make it m^2/), diffusivity air and acrylic

# k_ai, k_ac = 0.024, 0.19 # W/(m K), conductivity
k_ai, k_ac = 0.0, 0 # W/(m K), conductivity

L1_ai, L2_ai, L_ac = 0.03, 0.3, 0.05 # m, thickness of air layer or acrylic

N = 10000
n_x = 1000
n_t = 100
c = 60**2 / 10

t = np.linspace(0, 60**2*2, n_t)
T = np.exp(- t / c) * (20-6)+6


x1_ai, x2_ai, x_ac = np.linspace(0,L1_ai,n_x), np.linspace(0,L2_ai,n_x), np.linspace(0,L_ac,n_x)

sol1_ai = sol_heateq(x1_ai,t,T, N, diff_ai, k_ai/h_ai, L1_ai ) + np.expand_dims(T, axis=1)
sol2_ai = sol_heateq(x2_ai,t,T, N, diff_ai, k_ai/h_ai, L2_ai ) + np.expand_dims(T, axis=1)
sol_ac  = sol_heateq(x_ac ,t,T, N, diff_ac, k_ac/h_ac, L_ac  ) + np.expand_dims(T, axis=1)

name = '3cm of air'
plt.figure()
for i in range(0,n_t,int(n_t/40)):
    plt.plot( x1_ai, sol1_ai[i,:], c=[i/n_t,0,1-i/n_t], label=f'{t[i]:.2f}' )
    plt.plot([0,L1_ai], [T[i],20], 'b.')
# plt.legend()
plt.xlabel(r'$x$ (m)')
plt.ylabel(r'$T$ (°C)')
plt.title(name)
if save: plt.savefig(filepath+name+'.pdf', dpi=200, bbox_inches='tight')
plt.show()

name = '30cm of air'
plt.figure()
for i in range(0,n_t,int(n_t/40)):
    plt.plot( x2_ai, sol2_ai[i,:], c=[i/n_t,0,1-i/n_t], label=f'{t[i]:.2f}' )
    plt.plot([0,L2_ai], [T[i],20], 'b.')
plt.xlabel(r'$x$ (m)')
plt.ylabel(r'$T$ (°C)')
plt.title(name)
# plt.legend()
if save: plt.savefig(filepath+name+'.pdf', dpi=200, bbox_inches='tight')
plt.show()

name = '16cm of acrylic'
plt.figure()
for i in range(0,n_t,int(n_t/40)):
    plt.plot( x_ac, sol_ac[i,:], c=[i/n_t,0,1-i/n_t], label=f'{t[i]:.2f}' )
    plt.plot([0,L_ac], [T[i],20], 'b.')
plt.xlabel(r'$x$ (m)')
plt.ylabel(r'$T$ (°C)')
plt.title(name)
# plt.legend()
if save: plt.savefig(filepath+name+'.pdf', dpi=200, bbox_inches='tight')
plt.show()


#%%

calc = 1

diff_st = 10000 / 1e6 # mm^2/s (/1e6 to make it m^2/), diffusivity air and acrylic

k_st = 0 # W/(m K), conductivity

L_st = 0.0085 # m, thickness of air layer or acrylic

N = 10000
n_x = 1000
n_t = 100
c = 60**2 / 40

T0 = 21
tf = 13

t = np.linspace(0, 60 * 15, n_t)
T = np.exp(- t / c) * (T0-tf)+tf

T_til = T/T[0]

x_st = np.linspace(0,L_st,n_x)


plt.figure()
plt.plot( t/60 , T)
plt.xlabel('t (min)')
plt.ylabel('T(t) (°C)')
plt.grid()
plt.show()

if calc:
    sol_st = sol_heateq(x_st,t,T, N, diff_st, 0, L_st ) + np.expand_dims(T, axis=1)
    ener_d = energy_delayed(t, T_til, N, diff_st, 0, L_st )
    ener_b = energy_bath(t, T_til)

    plt.figure()
    plt.plot( t/60 , ener_d, label=r'$1-T-m(t)$')
    plt.plot( t/60 , ener_b, label=r'$1-T$')
    plt.plot( t/60 , ener_d / ener_b, label=r'Division')
    plt.xlabel('t (min)')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    name = '8.5mm of steel'
    plt.figure()
    for i in range(0,n_t,int(n_t/40)):
        plt.plot( x_ac, sol_st[i,:], c=[i/n_t,0,1-i/n_t], label=f'{t[i]:.2f}' )
        plt.plot([0,L_ac], [T[i],T0], 'b.')
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$T$ (°C)')
    plt.title(name)
    # plt.legend()
    # if save: plt.savefig(filepath+name+'.pdf', dpi=200, bbox_inches='tight')
    plt.show()





#%%











#%%











#%%











#%%






