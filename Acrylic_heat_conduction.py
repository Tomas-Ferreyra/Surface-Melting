#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 11:56:09 2026

@author: tomasferreyrahauchar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, fftconvolve
from time import time

def exp_convolution(T, t, beta):
    """
    Compute convolution of exp(-beta t) with signal T(t).

    Parameters
    ----------
    T : array
        Sampled signal T(t)
    t : array
        Time vector (uniformly spaced)
    beta : float
        Positive decay constant

    Returns
    -------
    y : array
        Convolution result (same length as T)
    """
    dt = t[1] - t[0]
    kernel = np.exp(-beta * t)
    y = convolve(T, kernel, mode='full') * dt
    return y[:len(T)]


# def bn(t,T,beta, gamma):
#     t1 = (0 - gamma * T[0]) * np.exp(-beta*t)
#     t2 = gamma * T
    
#     conv = exp_convolution(T, t, beta)
#     t3 = - gamma * beta * conv
#     return t1+t2+t3

# def un(x,t,n,T,beta,gamma, L):
#     b = np.expand_dims(bn(t, T, beta, gamma),axis=1)
#     sino = np.sin( n*np.pi/L * x)

#     return b * sino
    
# def t_sol(x,t,u,T, L):
#     v = (1 - x/L) * ( np.expand_dims(T,axis=1) - T[0] ) + T[0]
#     return u + v

def bn_gpt1(t,dT,beta, gamma):
    conv = exp_convolution(dT, t, beta)
    t3 = gamma * conv
    return t3

def u_gpt1(x,t,n,dT,beta,gamma,L):
    b = np.expand_dims(bn_gpt1(t,dT,beta, gamma),axis=1)
    sino = np.sin( n*np.pi/L * x)
    return b * sino

def t_sol1(x,t,u,T, L):
    v = (1 - x/L) * ( np.expand_dims(T,axis=1) ) + T[0] * x/L
    return v + u

    
def bn_gpt2(t, T, beta, alpha,L):
    b = exp_convolution(T-T[0], t, beta)
    return 2*alpha/L * b

def u_gpt2(x,t,n,T,beta,alpha,L):
    b = np.expand_dims( bn_gpt2(t,T,beta,alpha,L), axis=1)
    sino = np.sin( n*np.pi/L * x) * (-1)**(n+1)
    return b * sino

def t_sol2(x,t,u,T, L):
    return T[0] + u


from scipy.integrate import quad

# def heat_solution(x, t, P, alpha, L, N=50):
#     """
#     Compute solution of heat equation:
#         T_t = alpha T_xx
#     with:
#         T(x,0) = T0
#         T(0,t) = P(t)
#         T(L,t) = T0
#         P(0) = T0

#     Parameters
#     ----------
#     x : float
#     t : float
#     P : function  -> boundary function P(t)
#     alpha : float -> thermal diffusivity
#     L : float     -> rod length
#     N : int       -> number of Fourier terms

#     Returns
#     -------
#     T(x,t) : float
#     """

#     T0 = P[0]

#     # Steady linear part
#     steady = np.expand_dims(P(t))*(1 - x/L) + T0*(x/L)

#     # Series correction
#     series_sum = 0.0

#     for n in range(1, N+1):
#         lambda_n = alpha * (n*np.pi/L)**2

#         # Integrand for convolution term
#         integrand = lambda s: np.exp(-lambda_n*(t - s)) * \
#                               (np.gradient([P(s - 1e-8), P(s + 1e-8)],
#                                            2e-8)[1])

#         integral, _ = quad(integrand, 0, t)

#         term = (2/(n*np.pi)) * np.sin(n*np.pi*x/L) * integral
#         series_sum += term

#     return steady - series_sum


# def heat_solution_grid(x, t, P, alpha, L, N=50):
#     """
#     Returns T(t_i, x_j) as a 2D array of shape (len(t), len(x))
#     """

#     x = np.asarray(x)
#     t = np.asarray(t)

#     Nx = len(x)
#     Nt = len(t)

#     T = np.zeros((Nt, Nx))

#     T0 = P(0)

#     for i, ti in enumerate(t):

#         # steady part
#         steady = P(ti)*(1 - x/L) + T0*(x/L)

#         series_sum = np.zeros_like(x)

#         for n in range(1, N+1):
#             lambda_n = alpha * (n*np.pi/L)**2

#             integral, _ = quad(
#                 lambda s: np.exp(-lambda_n*(ti - s)) * P(s),
#                 0, ti
#             )

#             conv_term = (
#                 P(ti)
#                 - np.exp(-lambda_n*ti)*T0
#                 - lambda_n * integral
#             )

#             series_sum += (2/(n*np.pi)) * \
#                           np.sin(n*np.pi*x/L) * conv_term

#         T[i, :] = steady - series_sum

#     return T

# def heat_solution_stable(x, t, P, alpha, L, N=50):
#     """
#     Stable time-stepping Fourier solution.
#     Returns T(t_i, x_j) with shape (len(t), len(x))
#     """

#     x = np.asarray(x)
#     t = np.asarray(t)

#     Nx = len(x)
#     Nt = len(t)

#     T = np.zeros((Nt, Nx))

#     T0 = P(0)

#     # Precompute spatial modes
#     modes = np.array([
#         np.sin(n*np.pi*x/L)
#         for n in range(1, N+1)
#     ])

#     lambdas = np.array([
#         alpha*(n*np.pi/L)**2
#         for n in range(1, N+1)
#     ])

#     b = np.zeros(N)   # modal coefficients

#     for i in range(Nt):

#         if i > 0:
#             dt = t[i] - t[i-1]
#             dP = (P(t[i]) - P(t[i-1])) / dt

#             # implicit Euler (stable)
#             b = (b - dt*(2/(np.arange(1,N+1)*np.pi))*dP) \
#                 / (1 + dt*lambdas)

#         steady = P(t[i])*(1 - x/L) + T0*(x/L)

#         series = np.sum(
#             (2/(np.arange(1,N+1)*np.pi))[:,None] *
#             modes *
#             b[:,None],
#             axis=0
#         )

#         T[i,:] = steady - series

#     return T



def heat_solution(x, t, A, T0, alpha, L, Nmodes=50):
    """
    Compute T(x,t) for the 1D heat equation with:
        T(x,0)=T0
        T(0,t)=A(t)
        T(L,t)=T0

    Parameters
    ----------
    x : float or 1D array
        Spatial position(s)
    t : 1D array
        Time grid (uniformly spaced)
    A : 1D array
        Boundary temperature A(t) at x=0
    T0 : float
        Initial/background temperature
    alpha : float
        Diffusion coefficient
    L : float
        Domain length
    Nmodes : int
        Number of Fourier modes

    Returns
    -------
    T : array (len(x), len(t)) if x array,
        or (len(t),) if x scalar
    """
    
    t = np.asarray(t)
    A = np.asarray(A)
    dt = t[1] - t[0]
    
    f = A - T0
    x = np.atleast_1d(x)
    
    Nx = len(x)
    Nt = len(t)
    
    T = np.zeros((Nx, Nt))
    
    for n in range(1, Nmodes + 1):
        lam = alpha**2 * (n*np.pi/L)**2
        
        # Fast exponential convolution via recursion
        y = np.zeros(Nt)
        decay = np.exp(-lam * dt)
        coeff = (1 - decay) / lam
        
        for k in range(Nt - 1):
            y[k+1] = decay * y[k] + coeff * f[k]
        
        spatial = (2*alpha/L) * ((-1)**(n+1)) * np.sin(n*np.pi*x/L)
        
        T += spatial[:, None] * y[None, :]
    
    T += T0
    
    if T.shape[0] == 1:
        return T[0]
    
    return T

    
#%%
n_t = 100
n_x = 1000
alpha = 1
L = 1
T0 = 20

x = np.linspace(0,L,n_x)
t = np.linspace(0,10,n_t)

T = np.exp(-t) * (T0-6) + 6
dT = -np.exp(-t) * (T0-6)

plt.figure()
# plt.plot(t,T)

i = 1
plt.plot([x[0],x[-1]], [T[i],T[0]], '.')


# u_n = np.zeros( (len(t),len(x)) )
# for n in range(1,50):
    
#     beta = alpha * (n*np.pi/L)**2
#     gamma = -2 / (n*np.pi)
        
#     # b = bn(t, T, beta, gamma)
#     # u = u_gpt1(x, t, n, dT, beta, gamma, L) 
#     u = u_gpt2(x, t, n, dT, beta, alpha, L) 

#     u_n += u 
    
#     T_sol = t_sol2(x, t, u_n, T, L)

#     # plt.plot(t,u[:,i], '--', label=n)    
#     # plt.plot(x,u[i,:], '--', label=n)
#     # plt.plot(x,u_n[i,:], '--', label=n)
    
# plt.plot(x,T_sol[i,:], '--', label=n)

# plt.legend()
# plt.show()

plt.figr


#%%
    
n_t = 100
n_x = 1000
alpha = 0.01
L = 1
T0 = 20

x = np.linspace(0,L,n_x)
t = np.linspace(0,10,n_t)

A = np.exp(-t) * (T0-6) + 6

# T = np.exp(-t) * (T0-6) + 6
# dT = -np.exp(-t) * (T0-6)

solt = heat_solution(x, t, A, T0, alpha, L)
# solt = heat_solution_stable(x, t, P, alpha, L, N=10)

# plt.figure()

# for i in [0,1,2,3,10,50]:
    
    # plt.plot([x[0],x[-1]], [A[i],A[0]], 'b.')
    
    # plt.plot(t, solt[i,:], '-' )
    # plt.plot(x, solt[:,i], '-' )

# plt.imshow( solt )
# plt.show()


#%%


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

def bt_n(n,t,T,beta):
    ct = exp_convolution(t, T, beta)
    return 2/(n*np.pi) * (T[0] * np.exp(-beta*t) - T + beta * ct)

def un(n,x,t,T,beta,L):
    b = np.expand_dims( bt_n(n, t, T, beta), axis=1)
    sino = np.sin( n*np.pi/L * x)

    return b * sino
    
def t_sol(x,t,u,T, L):
    v = (1 - x/L) * ( np.expand_dims(T,axis=1) - T[0] ) + T[0]
    return u + v

def ener_integral(t,T,beta, L, N=50):
    b = 0
        
    for n in range(1,N):
        beta = alpha * (n*np.pi/L)**2
        
        nb = bt_n(2*n-1, t, T, beta)
        b += 2/(2*n-1) / np.pi * nb
        
    # for n in range(1,N):
    #     beta = alpha * (n*np.pi/L)**2
        
    #     nb = bt_n(n, t, T, beta)
    #     b += (1-(-1)**n) / (2*n-1) / np.pi * nb

    # return b  + (T+T[0])/2,(T+T[0])/2
    return b , (T+T[0])/2

n_x = 200
n_t = 300
alpha = 0.001
L = 1
T0 = 20

N = 1000

t = np.linspace(0, 6, n_t)
x = np.linspace(0, L, n_x)

T = np.exp(-t) * (T0-6)+6


utot = 0
# plt.figure()
for n in range(1,N):
    beta = alpha * (n*np.pi/L)**2
    
    btn = bt_n(n, t, T, beta) 
    u = un(n, x, t, T, beta, L)
    utot += u
    
#     if n <10: plt.plot( t, btn , label=n)
# plt.legend()
# plt.show()

twall = t_sol(x, t, utot, T, L)


plt.figure()
for i in range(0,n_t,int(n_t/10)):
# for i in range(0,10,1):
    # plt.plot( x, utot[i,:],label=f'{t[i]:.2f}'  )
    
    plt.plot( x, twall[i,:],label=i )
    # plt.plot( x, twall[i,:] - utot[i,:],label=i )
    # plt.plot([0,L], [T[i],T0], 'b.')

plt.legend()
plt.show()


gt,gx = np.gradient(twall, t,x, edge_order=2)
_,gxx = np.gradient(gx, t,x, edge_order=2)

# plt.figure()
# plt.imshow(gt)
# plt.colorbar()
# plt.show()
# plt.figure()
# plt.imshow(gxx)
# plt.colorbar()
# plt.show()

plt.figure()
plt.imshow(np.log( np.abs(gt - alpha*gxx)) ) 
plt.colorbar()
plt.show()


energy, tme = ener_integral( t, T, beta, L, N=int(N/2)) 
# energy = ener_integral( t, T, beta, N=int(N/2))

# enerin = np.trapezoid(twall, x, axis=-1)
enerin = np.trapezoid(utot, dx=x[1]-x[0], axis=-1)


plt.figure()
# plt.plot(t, energy + tme, label='Total energy')

# plt.plot(t, tme, label='T mean')

plt.plot(t, energy * L, '.-', label='b(t)')

plt.plot(t, enerin, label='Numerical')

plt.legend()
plt.show()

print( np.max(energy*L - enerin) )


#%%

plt.figure()

e1 = ener_integral(t, T, beta, N=1)
for n in range(2,10):
    energy = ener_integral( t, T, beta, N=n)
    # plt.plot(t, energy, label=n)
    plt.plot(t, energy - e1, label=n)
    print( np.max(np.abs(energy - e1 )) )

    e1 = energy     

          
plt.legend()
plt.show()









#%%




