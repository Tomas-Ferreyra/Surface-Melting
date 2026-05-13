#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 12:57:18 2025

@author: tomasferreyrahauchar
"""

import os
import re
import numpy as np 
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, least_squares
from scipy.stats import linregress
from scipy.integrate import cumulative_trapezoid

def solution_R( R, A,B ):
    pterm = A*R/B
    prod = (A+B) / (6 * np.cbrt(B-1)**2 * np.cbrt(B)**4)
    arcterm = 2*np.sqrt(3) * np.arctan( (1+2*R * np.cbrt(B/(B-1)) ) / np.sqrt(3) )
    logterm = np.log( ( (np.cbrt(B-1) + np.cbrt(B)*R)**2 - np.cbrt(B-1) * np.cbrt(B) * R ) / (np.cbrt(B-1) - np.cbrt(B)*R )**2  )
    return pterm + prod * (arcterm + logterm)

# def solution_T( R, A,B,Tm,To ):
#     u = 1 - R**3
#     top = (Tm*A + B*(To-Tm)) * u + To
#     bot = A*u + 1
#     return top/bot

def R_of_T(T, A,B, set_tot_cero=False):
    bot =  B + A*T
    argum = 1 - (1-T)/bot 
    if set_tot_cero:
        argum[argum<0] = 0
    return  np.cbrt(argum)

def V_of_T(T, A,B, set_tot_cero=False):
    bot = B + A*T
    argum = 1 - (1-T)/bot 
    if set_tot_cero:
        argum[argum<0] = 0
    return  argum 

def V_eloss_term(t,T, Tm,rhoi,Vo,L,cp,m,b):
    integrand = m * T + b
    bot = rhoi * Vo * (L + cp*(T-Tm)) 
    eloss = cumulative_trapezoid(integrand, t, initial=0)
    return eloss / bot

def T_adim(R,A,B):
    bot = A * (R**3-1) - 1
    top = (B+A) * (R**3-1)        
    return 1 - top/bot

def constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L ):
    beta = rhoi / rhow
    gamma = Vb/Vo
    Ste = L / (cp * (To-Tm))

    A = 2*beta/ gamma
    B = beta/gamma * (Ste-1)

    return A,B, beta,gamma,Ste

def constants_old( To, Tm, Vo, Vb, rhoi, rhow, cp, L ):
    beta = rhoi / rhow
    gamma = Vb/Vo
    Ste = L / (cp * (To-Tm))

    A = beta/gamma
    B = beta/gamma * Ste

    return A,B, beta,gamma,Ste

def lin_cons(t, val):
    tlin = t * val[0]
    tcons = np.ones_like(t) * val[1]
    funcion = np.minimum(tlin,tcons)
    return funcion

# Build color map for each label (folder + freq)
def get_color(folder_name, freq):
    if folder_name not in folder_colors:
        return "gray"
    cmap = folder_colors[folder_name]
    freq_index = frequencies.index(freq)
    return cmap(0.3 + 0.7 * freq_index / (len(frequencies) - 1))  # soft to dark

# === Function Definitions ===
def load_temperature_csv(file_path):
    df = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1", header=0)
    df['Timestamp'] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d_%H.%M.%S.%f")
    df.sort_values(by="Timestamp", inplace=True)
    df.set_index("Timestamp", inplace=True)
    t = (df.index - df.index[0]).total_seconds()
    return t.values, df["Water top °C"].values

def get_post_drop_segment(t, T, drop_threshold=0.1):
    T_initial = np.mean(T[t < 10])
    drop_indices = np.where(T <= T_initial - drop_threshold)[0]
    if drop_indices.size == 0:
        raise ValueError("No temperature drop found.")
    drop_idx = drop_indices[0]
    start_idx = drop_idx
    return t[start_idx:] - t[start_idx], T[start_idx:], T_initial

def split_into_windows(data, window_size):
    num_windows = len(data) // window_size
    return [data[i * window_size : (i + 1) * window_size] for i in range(num_windows)]

def average_per_window(data, window_size):
    return np.array([np.mean(w) for w in split_into_windows(data, window_size)])

def downsample_time(t, window_size):
    return np.array([np.mean(w) for w in split_into_windows(t, window_size)])

# Sort labels by folder then frequency
def sort_key(label):
    folder, freq = label.split()
    return (folder, frequencies.index(freq))


#%%

file_path = '/Volumes/ICESTOCKS/Ice Stocks/Melting/Test1/measures/after-im-test1-4Hz.csv'
df = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1", header=0)

# === Timestamp parsing ===
df['Timestamp'] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d_%H.%M.%S.%f")
df = df.sort_values(by="Timestamp")
df.set_index("Timestamp", inplace=True)


# === Setup ===
folder_paths = [
    '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Test7-5kg-experiment/Temperature Recordings',
    '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Test5-10kg-experiment/Temperature Recordings',
    '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Test6-20kg/Temperature Recordings',
]

# Define which frequencies exist in which folder
available_frequencies = {
    "Test7-5kg-experiment": ["1Hz", "2Hz", "4Hz", "8Hz", "12Hz"],
    "Test5-10kg-experiment": ["1Hz", "2Hz", "4Hz", "8Hz", "12Hz"],
    "Test6-20kg": ["2Hz", "4Hz", "8Hz"],
}

frequencies = ["1Hz", "2Hz", "4Hz", "8Hz", "12Hz"]

# Folder color bases
folder_colors = {
    "Test7-5kg-experiment": cm.Blues,
    "Test5-10kg-experiment": cm.Greens,
    "Test6-20kg": cm.Reds
}


# === Filter & Windowing Settings ===
window_size = 50
apply_savgol_filter = False
savgol_window = 25
savgol_polyorder = 2

# === Scan Folders for Matching Files (robust & unique per frequency) ===
file_paths = {}
for folder in folder_paths:
    folder_name = os.path.basename(os.path.dirname(folder)).strip()
    freqs = available_frequencies.get(folder_name)
    if freqs is None:
        print(f"⚠️ Skipping unknown folder: {folder_name}")
        continue

    print(f"📁 Scanning {folder_name}: expected {freqs}")
    seen_freqs = set()

    for filename in os.listdir(folder):
        filename_lower = filename.lower()
        for freq in freqs:
            pattern = rf'\b{freq.lower()}\b'  # match '2hz' as a whole word
            if re.search(pattern, filename_lower) and freq not in seen_freqs:
                label = f"{folder_name} {freq}"
                full_path = os.path.join(folder, filename)
                file_paths[label] = full_path
                seen_freqs.add(freq)
                print(f"  ✔ Found: {label}")
                break  # stop checking more frequencies for this file

# === Processing ===
results = {}

cut_at_min_rise = False # Toggle cut behavior
min_rise_threshold = 0.15  # °C above minimum

for label, path in file_paths.items():
    try:
        t, T_top = load_temperature_csv(path)

        # Segment raw data after initial drop
        t_seg, T_top_seg, T_top_init = get_post_drop_segment(t, T_top)

        # Downsample to averaged data
        t_avg = downsample_time(t_seg, window_size)
        T_top_avg = average_per_window(T_top_seg, window_size)

        # Match lengths
        min_len = min(len(t_avg), len(T_top_avg))
        valid_mask = ~np.isnan(T_top_avg)
        t_avg = t_avg[valid_mask]
        T_top_avg = T_top_avg[valid_mask]

        # === Cut after a single rise from minimum ===
        if cut_at_min_rise:
            min_index = np.argmin(T_top_avg)
            T_min = T_top_avg[min_index]

            # Find the first index where the temp rises above threshold
            above_thresh = np.where(T_top_avg[min_index + 1:] > T_min + min_rise_threshold)[0]

            if len(above_thresh) > 0:
                cut_index = min_index + 1 + above_thresh[0]  # get absolute index
                t_avg = t_avg[:cut_index]
                T_top_avg = T_top_avg[:cut_index]

        # Optional smoothing
        if apply_savgol_filter and len(T_top_avg) >= savgol_window:
            T_top_avg = savgol_filter(T_top_avg, window_length=savgol_window, polyorder=savgol_polyorder)

        # Store result
        results[label] = {
            "t_avg": t_avg,
            "T_top_avg": T_top_avg,
            "T_top_init": T_top_init,
        }

    except Exception as e:
        print(f"⚠️ Skipping {label} due to error: {e}")

#%%
rhow, rhoi = 998.2, 916.8 # kg/m3
Tm = 0 #°C
L = 334000 # J/kg 
cp = 4184 # J/(kg K)

model = 'old'

if   model == 'new': masses = {5:117, 10:112, 20:102} 
elif model == 'old': masses = {5:117, 10:112, 20:102} # To work with old {5:126, 10:123, 20:119}

plt.figure(figsize=(14, 8))

betas, gammas, stes, mints, minvs = [],[],[],[],[]
for i, label in enumerate(sorted(results.keys(), key=sort_key)):
    folder_name, freq = label.split()
    data = results[label]

    t = data["t_avg"]
    T = data["T_top_avg"]
    T_init = data["T_top_init"]

    mass = float(re.split('-| ',label)[1][:-2])
    frequ = int(re.split(' ',label)[1][:-2])
    
    Vo = mass / rhoi # m3
    # if mass == 20: Vo = 17 / rhoi
    To = T_init #°C
    Vb = masses[mass]/rhow #0.102 # m3        

    # A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )    
    # minv = np.min( V_of_T((T-Tm)/(To-Tm),A,B, set_tot_cero=False) )
    # if minv < 0:
    #     Vo = (1 - minv) * Vo

    if   model == 'new': A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
    elif model == 'old': A,B, beta,gamma,Ste = constants_old( To, Tm, Vo, Vb, rhoi, rhow, cp, L )

    # if 8<mass<15:
    #     print(f'{mass}kg, \t {frequ}Hz, \t A = {A:.4f}, \t B={B:.4f}')
    #     print(Vo, Vb )
    #     print()

    T_tilde = (T-Tm)/(To-Tm)

    print(fr"{mass:.0f} kg, {frequ} Hz, (beta,gamma,Ste,minT): ({beta:.4f},{gamma:.4f},{Ste:.4f},{np.min(T_tilde)})" )
    betas.append(beta)
    gammas.append(gamma)
    stes.append(Ste)
    mints.append(np.min(T_tilde))
    minvs.append( np.min(V_of_T(T_tilde,A,B, set_tot_cero=False)) )
    
    # Plot main temperature line
    # plt.plot(t, T,
    # plt.plot(t, T_tilde,
    plt.plot(t, V_of_T(T_tilde,A,B, set_tot_cero=False),
    # plt.plot(t, (1-V_of_T(T_tilde,A,B)) * Vo * rhoi,
             label=rf"{mass}kg, {frequ}Hz",
             color=get_color(folder_name, freq),
             linestyle='-',
               marker='o', markersize=8, markeredgecolor='black', linewidth=1)
              # marker='o', markersize=8)

    # --- Identify and plot the minimum point ---
    min_idx = np.argmin(T)
    t_min = t[min_idx]
    T_min = T[min_idx]
    
    Tsv = savgol_filter(T, len(T)//6, 3)

    # Plot star marker at minimum
    # plt.plot(t, Tsv, 'y--')
    # plt.plot(t, V_of_T((Tsv-Tm)/(To-Tm),A,B), 'y--')
    
    # plt.plot(t_min, T_min, marker='*', color='yellow',
    #          markersize=14, markeredgecolor='black', linewidth=1, zorder=5, label=None)

    # --- Print the minimum to terminal ---
    # print(f"⭐ {label}: Min Temperature = {T_min:.2f}°C at {t_min:.1f} s")

fsize = 15
plt.xlabel(r"$t$ (s)", fontsize=fsize)
# plt.ylabel("Top Temperature (°C)", fontsize=fsize)
plt.ylabel(r"$\tilde{V}$ (new)", fontsize=fsize)
# plt.title("Top Water Temperature vs Time (Minima Marked)", fontsize=fsize)
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Folder + Frequency", fontsize=fsize, ncols=1)
plt.tight_layout()

# filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/volume_old.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()

#%%
rhow, rhoi = 998.2, 916.8 # kg/m3
Tm = 0 #°C
L = 334000 # J/kg 
cp = 4184 # J/(kg K)

model = 'old'

R = np.linspace( 0,1, 1000 )


cutoffs = {5:0.5, 10:0.61, 20:1.1}
urmss = {1:0.003957, 2:0.00817, 4:0.01844, 8:0.03881, 12:0.06663}

if   model == 'new': masses = {5:117, 10:112, 20:102} 
elif model == 'old': masses = {5:124, 10:121, 20:119} # To work with old {5:126, 10:123, 20:119}

# 5 kg - 4Hz

mass_fig = [5,10,20]
# mass_fig = [5]

cs, cfs, freqs, mss, tinis = [],[],[],[], []
cerr = []

ratio = 1.5
plt.figure( figsize=(14/ratio,8/ratio) ) #figsize=(14, 8))

for i, label in enumerate(sorted(results.keys(), key=sort_key)):
    
    folder_name, freq = label.split()
    data = results[label]

    # --- Calcualte theoretical solution ---
    mass = float(re.split('-| ',label)[1][:-2])
    frequency = int(re.split(' ',label)[1][:-2])

    if mass in mass_fig:
        t = data["t_avg"]
        T = data["T_top_avg"]
        T_init = data["T_top_init"]
        
        print(f'{mass}kg, {frequency}Hz', end=' ')
        
        Vo = mass / rhoi # m3
        To = T_init #°C
        Vb = masses[mass]/rhow #0.102 # m3
        if   model == 'new': A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
        elif model == 'old': A,B, beta,gamma,Ste = constants_old( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
        Pr = 7
        
        # minv= np.min( V_of_T((T-Tm)/(To-Tm),A,B, set_tot_cero=False))
        # if minv < -0:
        #     Vo = (1 - minv) * Vo
        # print( '\t Real mass: {:.4f}kg,'.format(Vo * rhoi), end=' ' )
        
        # A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
        
        # ctheory = 0.001 * Ste * Pr / beta * urmss[frequency] / np.cbrt(Vo)
        
        # --- Identify and plot the minimum point ---
        min_idx = np.argmin(T)
        t_min = t[min_idx]
        T_min = T[min_idx]
    
        Tsv = savgol_filter(T, len(T)//6, 3)
    
        # mask = t<t_min      
        T_tilde = (T-Tm)/(To-Tm)
        V_tilde = V_of_T(T_tilde, A, B)
        
        Vsv = savgol_filter(V_tilde, len(V_tilde)//4, 3)
        mask1 = np.gradient(Vsv)>0
        mask2 = Vsv<0.05
        if np.sum(mask2)>0:
            fin = np.min( [np.where(mask1)[0][0], np.where(mask2)[0][0] ]) 
        else:
            fin = np.where(mask1)[0][0]
            
        # fin = len(t)

        fitfun = lambda Temp,c: np.real( ( solution_R( R_of_T(Temp, A, B, set_tot_cero=False) , A, B) - solution_R(1, A, B) ) ) / c
        
        def funfit(val):
            tlin = t * val[0]
            tcons = np.ones_like(t) * val[1]
            funcion = np.minimum(tlin,tcons)
            return np.abs( T_linear - funcion )
        
        def funfitf(val):
            tlin = t[:fin] * val[0]
            tcons = np.ones_like(t[:fin]) * val[1]
            funcion = np.minimum(tlin,tcons)
            return np.abs( T_linear[:fin] - funcion )
        
            
        # Plot main temperature line using function fitfun (should be linear some part)
        T_linear = fitfun(T_tilde,1)
        Tsv_linear = fitfun((Tsv-Tm)/(To-Tm),1)
        
        
        mask = t < fin
        constant_sl = lambda T,c: c*T
        cc,cov = curve_fit(constant_sl, t[mask], T_linear[mask] )
        c = cc[0]
        # print(label,'kg, c = {:.4f}'.format(c) )
        # print('\t No ice:{:.3f}, {:.1f}% of ice: {:.3f}'.format(solution_R(1, A, B) - solution_R(0, A, B), 0.55**3 * 100 ,solution_R(1, A, B) - solution_R(0.55, A, B)))
        # # print('\t 7°C:{:.3f}, 12°C: {:.3f}'.format(solution_R(1, A, B) - solution_R(R_of_T( 7, A, B, Tm, To) , A, B),  \
        # #                                            solution_R(1, A, B) - solution_R(R_of_T(12, A, B, Tm, To), A, B)))

        # Plot main temperature line using function fitfun (should be linear some part)
        # T_linear = fitfun(T_tilde,1)
        ls = least_squares(funfit, [c,0.6])
        lsf = least_squares(funfitf, [c,10])
        c, cons = ls.x
        cf, consf = lsf.x

        print('c = {:.4f}, res = {:.4f}'.format(c, np.sum(ls.fun))) #, 'cf = {:.4f}'.format(cf) )
        # print('\t No ice:{:.3f}, {:.1f}% of ice: {:.3f}'.format(solution_R(1, A, B) - solution_R(0, A, B), 0.55**3 * 100 ,solution_R(1, A, B) - solution_R(0.55, A, B)))
        # print('\t 7°C:{:.3f}, 12°C: {:.3f}'.format(solution_R(1, A, B) - solution_R(R_of_T( 7, A, B, Tm, To) , A, B),  \
        #                                            solution_R(1, A, B) - solution_R(R_of_T(12, A, B, Tm, To), A, B)))

        frequ = int(re.split('-| ',label)[-1][:-2])
        cs.append( c )
        cfs.append( cf )
        freqs.append( frequ )
        mss.append( mass )
        tinis.append(T_init)
        # cerr.append( np.sqrt(cov[0,0]) )
    
    
        # plt.plot(t , T_linear, label=f'{frequ} Hz',
        plt.plot(t*c , T_linear, label=f'{frequ} Hz',
                  # label=rf"{mass}kg, {freq} ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                  color=get_color(folder_name, freq),
                  linestyle='-',
                  # marker='o', markersize=8, markeredgecolor='black', linewidth=1)
                  marker='o', markersize=8) 
        
        # plt.plot(t, Tsv_linear, 'y--')
        
        # plt.plot( t  , lin_cons(t, ls.x) , '--', color='darkviolet')
        plt.plot( t*c , lin_cons(t, ls.x) , 'm--' )
        
        # plt.plot(t,V_tilde,'.-')

 
# plt.hlines(cutoffs[mass_fig[0]], 0, 500, color='black', linestyles='dashed', label='Fit limit' ) #plot limit of fit 
# plt.plot([0,2],[0,2],'--',color='y')

fsize = 12
# plt.xlabel(r"$t$ (s)", fontsize=fsize )
plt.xlabel(r"$C t$", fontsize=fsize )
plt.ylabel(r"$R_s(R_T(\tilde{T})) - R_s(1)$", fontsize=fsize )
# plt.title(f"{mass_fig[0]} kg", fontsize=fsize )
plt.grid(True)

# plt.xscale('log')
# plt.yscale('log')

plt.legend(loc='lower right', ncols=3)
plt.tight_layout()


filename = f'/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/model_fit_{mass_fig[0]}kg.pdf'
# filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/model_all_mass_fitted.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')
# print()
# print(filename)

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Folder + Frequency")
plt.show()

#%%

cs = np.array(cs)
cfs = np.array(cfs)
cerr = np.array(cerr)
mss = np.array(mss)
freqs = np.array(freqs) 
tinis = np.array(tinis) 

mass_fig = [5,10,20]

plt.figure()
for i, label in enumerate(sorted(results.keys(), key=sort_key)):
    
    folder_name, freq = label.split()
    data = results[label]

    # --- Calcualte theoretical solution ---
    mass = float(re.split('-| ',label)[1][:-2])

    if mass in mass_fig:
        t = data["t_avg"]
        T = data["T_top_avg"]
        T_init = data["T_top_init"]
        
        
        Vo = mass / rhoi # m3
        To = T_init #°C
        Vb = masses[mass]/rhow #0.102 # m3
        
        if   model == 'new': A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
        elif model == 'old': A,B, beta,gamma,Ste = constants_old( To, Tm, Vo, Vb, rhoi, rhow, cp, L )


    Tf = T_adim(0, A, B)

    plt.plot( t * cs[i], ( (T-Tm)/(To-Tm) ), 
            # label=rf"{mass}kg, {freq} ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                color=get_color(folder_name, freq),
                linestyle='-',
                # marker='o', markersize=8)
                marker='o', markersize=8, markeredgecolor='black', linewidth=1)
    
    R = np.linspace(1,0,1000)
    plt.plot(  solution_R(R,A,B) - solution_R(R[:1],A,B) , ( T_adim(R, A, B))   , 'c--', zorder=20 )

plt.show()


#%%
cs = np.array(cs)
cfs = np.array(cfs)
cerr = np.array(cerr)
mss = np.array(mss)
freqs = np.array(freqs) 
tinis = np.array(tinis) 

Pr = 7
umrss = np.array([0.003957, 0.00817, 0.01844, 0.03881, 0.06663])
color = [0,'blue','green',0,'red']

# rhow/rhoi*cp/L * umrss

fig, ax = plt.subplots(1,2, figsize=(10,4), layout="constrained")
for i in [5,10,20]:
    mask = mss==i
    ax[0].plot( freqs[mask], cs[mask] , 'o', label=f'{i} kg', color=color[i//5], markeredgecolor='k')
    # ax[0].plot( freqs[mask], cfs[mask] , 's', label=f'{i} kg', color=color[i//5], markeredgecolor='k')

    ax[1].plot( freqs[mask], cs[mask] * np.cbrt(i / rhoi) / tinis[mask]  , 'o', label=f'{i} kg', color=color[i//5], markeredgecolor='k')
    # ax[1].plot( freqs[mask], cfs[mask] * np.cbrt(i / rhoi) / tinis[mask]  , 's', label=f'{i} kg', color=color[i//5], markeredgecolor='k')

    # ax[0].errorbar( freqs[mask], cs[mask], yerr=cerr[mask] , fmt='o', label=f'{i} kg', color=color[i//5], markeredgecolor='k')
    # ax[1].errorbar( freqs[mask], cs[mask] * np.cbrt(i / rhoi) / tinis[mask], yerr=cerr[mask] * np.cbrt(i / rhoi) / tinis[mask] , \
    #               fmt='o', label=f'{i} kg', color=color[i//5], markeredgecolor='k')

ax[1].plot( [1,2,4,8,12], rhow/rhoi*cp/L * umrss / Pr * 0.75 , '^', color='orange', label='Theory',zorder=4)

ax[0].legend()
ax[1].legend()

ax[0].set_ylabel(r'$C$ (1/s)')
ax[0].set_xlabel(r'$f$ (Hz)')

ax[1].set_ylabel(r'$C \,\, V_0^{1/3} \, / \, \Delta T$ (m/sK)')
ax[1].set_xlabel(r'$f$ (Hz)')

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/constant_theory.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')


plt.show()

#%%
# =============================================================================
# Looking at varying inital mass or initial Vb
# =============================================================================

def fit_model(t, T_tilde, A, B, set_tot_cero=False, use_fin=False ):

    
    V_tilde = V_of_T(T_tilde, A, B, set_tot_cero=False)
    
    Vsv = savgol_filter(V_tilde, len(V_tilde)//4, 3)
    mask1 = np.gradient(Vsv)>0
    mask2 = Vsv<0.05
    if np.sum(mask2)>0:
        fin = np.min( [np.where(mask1)[0][0], np.where(mask2)[0][0] ]) 
    else:
        fin = np.where(mask1)[0][0]
        
    fitfun = lambda Temp,c: np.real( ( solution_R( R_of_T(Temp, A, B, set_tot_cero=False) , A, B) - solution_R(1, A, B) ) ) / c
    T_linear = fitfun(T_tilde,1)    
    
    def funfit(val):
        tlin = t * val[0]
        tcons = np.ones_like(t) * val[1]
        funcion = np.minimum(tlin,tcons)
        return np.abs( T_linear - funcion )
    
    def funfitf(val):
        tlin = t[:fin] * val[0]
        tcons = np.ones_like(t[:fin]) * val[1]
        funcion = np.minimum(tlin,tcons)
        return np.abs( T_linear[:fin] - funcion )
    
    # Plot main temperature line using function fitfun (should be linear some part)
    
    mask = t < fin
    constant_sl = lambda T,c: c*T
    cc,cov = curve_fit(constant_sl, t[mask], T_linear[mask] )
    c = cc[0]

    if use_fin: ls = least_squares(funfitf, [c,10])
    else: ls = least_squares(funfit, [c,0.9])

    return T_linear, ls


#%%
rhow, rhoi = 998.2, 916.8 # kg/m3
Tm = 0 #°C
L = 334000 # J/kg 
cp = 4184 # J/(kg K)

model = 'old'

R = np.linspace( 0,1, 1000 )

cutoffs = {5:0.5, 10:0.61, 20:1.1}
urmss = {1:0.003957, 2:0.00817, 4:0.01844, 8:0.03881, 12:0.06663}

if   model == 'new': masses = {5:117, 10:112, 20:102} 
if   model == 'old': masses = {5:117, 10:112, 20:102} 
# elif model == 'old': masses = {5:126, 10:123, 20:120} # To work with old {5:126, 10:123, 20:119}


# 5 kg - 4Hz

# mass_fig = [5,10,20]
mass_fig = [20]
freqs_fig = [8]

cs, cfs, freqs, mss, tinis = [],[],[],[], []
cerr = []

ratio = 1.5
# plt.figure( figsize=(14/ratio,8/ratio) ) 
fig, ax = plt.subplots(1,2, figsize=(18/ratio,8/ratio) ) 

for i, label in enumerate(sorted(results.keys(), key=sort_key)):
    
    folder_name, freq = label.split()
    data = results[label]

    # --- Calcualte theoretical solution ---
    mass = float(re.split('-| ',label)[1][:-2])
    frequency = int(re.split(' ',label)[1][:-2])

    if (mass in mass_fig) and (frequency in freqs_fig):
        t = data["t_avg"]
        T = data["T_top_avg"]
        T_init = data["T_top_init"]
        
        print(f'{mass}kg, {frequency}Hz')
        
        Vo = mass / rhoi # m3
        To = T_init #°C
        
        # in_watmas = masses[mass]
        # in_watmas = 90
        for in_watmas in [98,99,100,102,104]:
        # for in_watmas in [masses[mass]]:
            Vb = in_watmas /rhow #0.102 # m3
            
            print(f'Ini water mass: {in_watmas}kg', end= ' ' )
    
            if   model == 'new': A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
            elif model == 'old': A,B, beta,gamma,Ste = constants_old( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
            Pr = 7
            
            ctheory = 0.001 * Ste * Pr / beta * urmss[frequency] / np.cbrt(Vo)
            
            # --- Identify and plot the minimum point ---
            min_idx = np.argmin(T)
            t_min = t[min_idx]
            T_min = T[min_idx]
        
            # Tsv = savgol_filter(T, len(T)//6, 3)
        
            # mask = t<t_min      
            T_tilde = (T-Tm)/(To-Tm)
            V_tilde = V_of_T(T_tilde, A, B, set_tot_cero=False)
            
            # Tsv_t = savgol_filter(T_tilde, len(T_tilde)//6, 3)
            # Vsv_t = V_of_T(Tsv_t, A, B, set_tot_cero=False)

            T_linear, ls = fit_model(t, T_tilde, A, B)

            c, cons = ls.x
    
            print('c = {:.4f}, res = {:.4f}'.format(c, np.sum(ls.fun))) #, 'cf = {:.4f}'.format(cf) )
    
            frequ = int(re.split('-| ',label)[-1][:-2])
            cs.append( c )
            freqs.append( frequ )
            mss.append( mass )
            tinis.append(T_init)        
        
            ax[0].plot(t, T_linear,
                      # label=rf"{mass}kg, {freq} ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                      # color=get_color(folder_name, freq),
                      linestyle='-',
                      # marker='o', markersize=8, markeredgecolor='black', linewidth=1)
                      marker='o', markersize=8, label=f'{in_watmas} kg') 
            
            ax[0].plot( t, lin_cons(t, ls.x) , 'k--', zorder=10 )
            # ax[0].plot( t[:fin], lin_cons(t[:fin], lsf.x) , 'c--' )
            
            ax[1].plot(t,V_tilde,'.-', label=f'{in_watmas} kg')
            # ax[1].plot(t,Vsv_t,'y--', label=f'{in_watmas} kg')

 
# plt.hlines(cutoffs[mass_fig[0]], 0, 500, color='black', linestyles='dashed', label='Fit limit' ) #plot limit of fit 
# plt.plot([0,2],[0,2],'--',color='y')

fsize = 12
ax[0].set_xlabel(r"$t$ (s)", fontsize=fsize )
ax[0].set_ylabel(r"$R_s(R_T(T)) - R_s(1)$", fontsize=fsize )
# plt.title(f"{mass_fig[0]} kg", fontsize=fsize )
ax[0].grid(True)

ax[1].set_xlabel(r"$t$ (s)", fontsize=fsize )
ax[1].set_ylabel(r"$\tilde{V}$", fontsize=fsize )
ax[1].grid(True)
ax[1].set_ylim(-0.06,0.3)

# ax[0].set_xscale('log')
# ax[0].set_yscale('log')

ax[0].legend(loc='lower right')
ax[1].legend(loc='upper right')
plt.tight_layout()

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/different_Vb.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()

#%%

mass_fig = [20]
freqs_fig = [8]


cs, vbs, ctes, res = [],[],[],[]

for i, label in enumerate(sorted(results.keys(), key=sort_key)):
    
    folder_name, freq = label.split()
    data = results[label]

    # --- Calcualte theoretical solution ---
    mass = float(re.split('-| ',label)[1][:-2])
    frequency = int(re.split(' ',label)[1][:-2])

    if (mass in mass_fig) and (frequency in freqs_fig):
        t = data["t_avg"]
        T = data["T_top_avg"]
        T_init = data["T_top_init"]
        
        Vo = mass / rhoi # m3
        To = T_init #°C
        
        in_watmas = masses[mass]
        distib_watmas = np.random.normal(in_watmas, 2, 100000 )
        
        for in_watmas in tqdm(distib_watmas):
            Vb = in_watmas /rhow #0.102 # m3
            if   model == 'new': A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
            elif model == 'old': A,B, beta,gamma,Ste = constants_old( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
            Pr = 7
            
            T_tilde = (T-Tm)/(To-Tm)
            
            T_linear, ls = fit_model(t, T_tilde, A, B)
            # c, cons = ls.x[0], ls.x[1]
            
            cs.append( ls.x[0] )
            ctes.append( ls.x[1] )
            vbs.append( Vb * rhow )
            res.append( np.sum(ls.fun) )
     
cs, vbs, ctes, res = np.array(cs), np.array(vbs), np.array(ctes), np.array(res)  
#%%

fig,ax = plt.subplots(1,3, figsize=(15,5))
ax[0].hist( vbs, bins=100, density=True )
ax[1].hist( cs, bins=100, density=True )
ax[2].hist( res, bins=100, density=True )

ax[0].set_xlabel(r'$V_b \rho_w$ (kg)')
ax[1].set_xlabel(r'$C$ (1/s)')
ax[2].set_xlabel(r'Residue')
plt.tight_layout()

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/statistics.pdf'
plt.savefig(filename, dpi=200, bbox_inches='tight')
plt.show()

pi = 1-res/np.max(res)
pi = pi/np.sum(pi)

w_mean, u_mean = np.sum( pi*cs ), np.mean(cs)
w_std, u_std = np.sqrt( np.sum( pi*(cs-u_mean)**2 ) ), np.std(cs)
ind = np.argmin( np.abs(cs - w_mean) )
print( w_mean, u_mean, vbs[ind] )
print( w_std, u_std, vbs[ind] )


indsort = np.argsort(vbs)
skip = 10
fig,ax = plt.subplots(1,3, figsize=(15,5))
ax[0].plot( vbs[indsort][::skip], cs[indsort][::skip], '.' )
# ax[0].hlines( [w_mean,u_mean], np.min(vbs), np.max(vbs), linestyles='dashed', colors=['r','g']   )

ax[1].plot( vbs[indsort][::skip], res[indsort][::skip], '.' )
ax[2].plot( vbs[indsort][::skip], ctes[indsort][::skip], '.' )
ax[0].set_ylabel(r'$C$ (1/s)')
ax[1].set_ylabel(r'Residue')
ax[2].set_ylabel(r'Constant')
ax[0].set_xlabel(r'$V_b \rho_w$ (kg)')
ax[1].set_xlabel(r'$V_b \rho_w$ (kg)')
ax[2].set_xlabel(r'$V_b \rho_w$ (kg)')
plt.tight_layout()

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/cs_res_vbs.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')
plt.show()
# plt.close('all')




#%%
# =============================================================================
# Towards Nu vs Re
# =============================================================================
# Corrected model?

# def constantss( To, Tm, Vo, Vb, rhoi, rhow, cp, L ):
#     beta = rhoi / rhow
#     gamma = Vb/Vo    
#     Ste = L / (cp * (To-Tm))
        
#     return beta, gamma, Ste

# def V_of_T_bgs(T,To,Tm, beta,gamma,Ste ):
#     bot = beta * ( Ste*(To-Tm) + 2*T-Tm-To ) 
#     top = gamma * (T-To)
#     return 1 + top / bot

def V_eloss_term(t,T,To,Tm, beta,gamma,Ste, rhow,Vo,cp,m,b):
    bot = beta * ( Ste*(To-Tm) + 2*T-Tm-To ) 
    integrand = m * T + b
    top = cumulative_trapezoid(integrand, t, initial=0)
    return top / (bot * rhow * Vo * cp )


#%%
# Calculate R(t) using energy balance
rhow, rhoi = 998.2, 916.8 # kg/m3
Tm = 0 #°C
L = 334000 # J/kg 
cp = 4184 # J/(kg K)

#Energy loss fit parameters
m = -8.077026040191905
b = 188.25141687003767

plt.rcParams.update({'font.size':15})

model = 'old'

if   model == 'new': masses_bath = {5:117, 10:112, 20:102} 
elif model == 'old': masses_bath = {5:117, 10:112, 20:102} 
# elif model == 'old': masses_bath = {5:126, 10:123, 20:120} # To work with old {5:126, 10:123, 20:119}

apply_heat_loss = False
show_minima = False

sav_gol_fil = False
gradient = False

try_mass = [5,10,20]
# try_mass = [10]

#Plot Radius and Volume over time (or its derivatives over time)
fig, ax = plt.subplots(1,3,figsize=(22,7))

for i, label in enumerate(sorted(results.keys(), key=sort_key)):
    folder_name, freq = label.split()
    data = results[label]

    t = data["t_avg"]
    T = data["T_top_avg"]
    T_init = data["T_top_init"]

    frequency = re.split(' ',label)[1][:-2]
    mass = float(re.split('-| ',label)[1][:-2])
    Vo = mass / rhoi # m3
    To = T_init #°C
    Vb = masses_bath[mass] / rhow #0.102 # m3
    
    if mass not in try_mass:
        continue


    # beta,gamma,Ste = constantss( To, Tm, Vo, Vb, rhoi, rhow, cp, L )      
    # V = V_of_T_bgs(T, To,Tm, beta,gamma,Ste)
    
    if   model == 'new': A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
    elif model == 'old': A,B, beta,gamma,Ste = constants_old( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
    T_tilde = (T-Tm)/(To-Tm)
    V = V_of_T(T_tilde,A,B) 
    
    print(fr"{mass:.0f} kg, {frequency} Hz, $\beta$ {beta:.4f}, $\gamma$ {gamma:.4f}, Ste {Ste:.4f}" )

    if apply_heat_loss:
        V -= V_eloss_term(t,T,To,Tm, beta,gamma,Ste, rhow,Vo,cp,m,b)

    R = np.cbrt(V)
    
    if sav_gol_fil:
        Vsv = savgol_filter(V, len(V)//4, 3)
        mask1 = np.gradient(Vsv)>0
        mask2 = Vsv<0.05
        if np.sum(mask2)>0:
            fin = np.min( [np.where(mask1)[0][0], np.where(mask2)[0][0] ]) 
        else:
            fin = np.where(mask1)[0][0]
    else: fin = -1

    # Plot radius and volume over time (with this energy balance)
    if not gradient:
        ax[2].plot(t, (T-Tm)/(To-Tm),
                  label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                  color=get_color(folder_name, freq),
                  linestyle='-',
                  marker='o', markersize=8,
                  markeredgecolor='black', linewidth=1)
        
        ax[1].plot(t[:fin], R[:fin],
                  label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                  color=get_color(folder_name, freq),
                  linestyle='-',
                  marker='o', markersize=8,
                  markeredgecolor='black', linewidth=1)

        # if show_minima: ax[1].plot( t[fin], R[fin], marker='*', color='yellow',
        #           markersize=14, markeredgecolor='black', linewidth=1, zorder=5, label=None )
        if sav_gol_fil:
            ax[1].plot(t[:fin], np.cbrt(Vsv[:fin]),
                      # color=get_color(folder_name, freq),
                      color='y',
                      linestyle='--', linewidth=1)

        ax[0].plot(t, V,
                  label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                  color=get_color(folder_name, freq),
                  linestyle='-',
                  marker='o', markersize=8,
                  markeredgecolor='black', linewidth=1)
        
        if show_minima: ax[0].plot( t[fin], V[fin], marker='*', color='yellow',
                  markersize=14, markeredgecolor='black', linewidth=1, zorder=5, label=None )
        if sav_gol_fil:
            ax[0].plot(t, Vsv,
                      # color=get_color(folder_name, freq),
                      color='y',
                      linestyle='--', linewidth=1)

    # Plot gradient if radius and volume over time 
    else:
        ax[1].plot(t[:fin], np.gradient(np.cbrt(Vsv),t)[:fin],
                 label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                 color=get_color(folder_name, freq),
                 linestyle='-',
                 marker='o', markersize=8,
                 markeredgecolor='black', linewidth=1)
        ax[0].plot(t[:fin], np.gradient(Vsv,t)[:fin],
                 label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                  color=get_color(folder_name, freq),
                 # color='k',
                 linestyle='-',
                 marker='o', markersize=8,
                 markeredgecolor='black', linewidth=1)

# plt.xscale('log')
# plt.yscale('log')

fsize = 11
ax[0].tick_params(axis='both', which='major', labelsize=fsize)
ax[1].tick_params(axis='both', which='major', labelsize=fsize)
ax[2].tick_params(axis='both', which='major', labelsize=fsize)
ax[0].set_xlabel("t (s)", fontsize=fsize)
ax[1].set_xlabel("t (s)", fontsize=fsize)
ax[2].set_xlabel("t (s)", fontsize=fsize)
if not gradient:
    ax[0].set_ylabel(r"$\tilde{V}$ ", fontsize=fsize)
    ax[1].set_ylabel(r"$\tilde{R}$ ", fontsize=fsize)
    ax[2].set_ylabel(r"$\tilde{T}$ ", fontsize=fsize)
else:
    ax[1].set_ylabel(r"$d\tilde{V}/dt$ ", fontsize=fsize)
    ax[0].set_ylabel(r"$d\tilde{R}/dt$ ", fontsize=fsize)
    ax[2].set_ylabel(r"$\tilde{T}$ ", fontsize=fsize)

# ax[1].set_title("Radius vs Time", fontsize=fsize)
# ax[0].set_title("Volume vs Time", fontsize=fsize)

for i in range(3):
    ax[i].grid(True)
    ax[i].set_ylim(-0.02,1.02)

plt.tight_layout()
# h, l = ax[1].get_legend_handles_labels()
# plt.legend(h,l,loc='upper center', ncols=5, bbox_to_anchor=(0.5, -0.5), fancybox=False, shadow=False, fontsize=fsize)
ax[1].legend(loc='upper center', ncols=5, bbox_to_anchor=(0.5, 1.2), fancybox=False, shadow=False, fontsize=fsize)
fig.subplots_adjust(bottom=0.08, top=0.85)

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/experiments.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()

#%%

apply_heat_loss = False

# dR/dt vs Temperature
fig, ax = plt.subplots(1,2,figsize=(18,5))

for i, label in enumerate(sorted(results.keys(), key=sort_key)):
    folder_name, freq = label.split()
    data = results[label]

    t = data["t_avg"]
    T = data["T_top_avg"]
    T_init = data["T_top_init"]

    frequency = re.split(' ',label)[1][:-2]
    mass = float(re.split('-| ',label)[1][:-2])
    Vo = mass / rhoi # m3
    To = T_init #°C
    Vb = masses_bath[mass] / rhow #0.102 # m3
    
    # beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )      
    # V = V_of_T(T, To,Tm, beta,gamma,Ste)
    if   model == 'new': A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
    elif model == 'old': A,B, beta,gamma,Ste = constants_old( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
    T_tilde = (T-Tm)/(To-Tm)
    V = V_of_T(T_tilde,A,B) 

    if apply_heat_loss:
        V -= V_eloss_term(t,T,To,Tm, beta,gamma,Ste, rhow,Vo,cp,m,b)
        
    R = np.cbrt(V)

    Vsv = savgol_filter(V, len(V)//4, 3)
    mask1 = np.gradient(Vsv)>0
    mask2 = Vsv<0
    if np.sum(mask2)>0:
        fin = np.min( [np.where(mask1)[0][0], np.where(mask2)[0][0] ]) 
    else:
        fin = np.where(mask1)[0][0]

    Tsv = savgol_filter(T, 40, 3)
    Rsv = np.cbrt(Vsv)


    ax[1].plot(Tsv[:fin], np.gradient(Rsv,t)[:fin],
             label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
             color=get_color(folder_name, freq),
             linestyle='-',
             marker='o', markersize=8,
             markeredgecolor='black', linewidth=1)
    ax[0].plot(t[:fin], np.gradient(Rsv,t)[:fin],
             label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
             color=get_color(folder_name, freq),
             linestyle='-',
             marker='o', markersize=8,
             markeredgecolor='black', linewidth=1)

# plt.xscale('log')
# plt.yscale('log')

fsize = 15
ax[0].tick_params(axis='both', which='major', labelsize=fsize)
ax[1].tick_params(axis='both', which='major', labelsize=fsize)
ax[0].set_xlabel(r"$t$ (s)", fontsize=fsize)
ax[1].set_xlabel(r"$T$ (°C)", fontsize=fsize)
ax[1].set_ylabel(r"$d\tilde{R}/dt$ ", fontsize=fsize)
ax[0].set_ylabel(r"$d\tilde{R}/dt$ ", fontsize=fsize)
ax[0].set_title("Radius vs Time", fontsize=fsize)
ax[1].set_title("Radius vs Temperature", fontsize=fsize)

ax[0].grid(True)
ax[1].grid(True)
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Mass + Frequency", fontsize=fsize)
plt.tight_layout()

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/temperatures.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()

#%%

# Nu and Re (per time and experiment)
rhow, rhoi = 998.2, 916.8 # kg/m3
Tm = 0 #°C
L = 334000 # J/kg 
cp = 4184 # J/(kg K)
nu = 1.0035e-6 #m2/s ,dynamic viscosity at 20°C
kappa = 0.143e-6 #m2/s ,dynamic viscosity at 20°C

Pr = nu/kappa

model = 'old'

compensate = 1

umrss = {1:0.003957, 2:0.00817, 4:0.01844, 8:0.03881, 12:0.06663} #m/s, u_rms

if   model == 'new': masses_bath = {5:117, 10:112, 20:102} 
elif model == 'old': masses_bath = {5:117, 10:112, 20:102} 
# elif model == 'old': masses_bath = {5:126, 10:123, 20:120} # To work with old {5:126, 10:123, 20:119}

# fig, ax = plt.subplots(1,2,figsize=(14,8) )
fig, ax = plt.subplots(1,2,figsize=(15,5), sharey=False )

for i, label in enumerate(sorted(results.keys(), key=sort_key)):
    folder_name, freq = label.split()
    data = results[label]

    t = data["t_avg"]
    T = data["T_top_avg"]
    T_init = data["T_top_init"]

    frequency = re.split(' ',label)[1][:-2]
    mass = float(re.split('-| ',label)[1][:-2])
    Vo = mass / rhoi # m3
    To = T_init #°C
    Vb = masses_bath[mass] / rhow #0.102 # m3
    u_rms = umrss[int(frequency)]
    
    # beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )      
    # V = V_of_T(T, To,Tm, beta,gamma,Ste)
    if   model == 'new': A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
    elif model == 'old': A,B, beta,gamma,Ste = constants_old( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
    T_tilde = (T-Tm)/(To-Tm)
    V = V_of_T(T_tilde,A,B) 

    if apply_heat_loss:
        V -= V_eloss_term(t,T,To,Tm, beta,gamma,Ste, rhow,Vo,cp,m,b)
        
    R = np.cbrt(V)

    Vsv = savgol_filter(V, len(V)//4, 3)
    mask1 = np.gradient(Vsv)>0
    mask2 = Vsv<0.3
    if np.sum(mask2)>0:
        fin = np.min( [np.where(mask1)[0][0], np.where(mask2)[0][0] ]) 
    else:
        fin = np.where(mask1)[0][0]

    Tsv = savgol_filter(T, 40, 3)
    Rsv = np.cbrt(Vsv)
    gRsv = np.gradient(Rsv,t)
    
    Nu = - beta * Ste * Pr * np.mean(gRsv[:fin]) * Rsv[0] * np.cbrt(Vo)**2 / nu
    Re = u_rms * Rsv[0] * np.cbrt(Vo) / nu
    
    Ste_t = L / (cp * (T-Tm))
    Nut = - beta * Ste_t[:fin] * Pr * gRsv[:fin] * Rsv[:fin] * np.cbrt(Vo)**2 / nu
    # Nut = - beta * Ste * Pr * gRsv[:fin] * Rsv[:fin] * np.cbrt(Vo)**2 / nu
    Ret = u_rms * Rsv[:fin] * np.cbrt(Vo) / nu
    
    if compensate == 0:
        ax[0].scatter(Re, Nu , 
                      label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                      color=get_color(folder_name, freq),
                      linestyle='-',
                      marker='o', s=30,
                      edgecolors='black', linewidth=1)
        ax[1].scatter(Ret, Nut , 
                      label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                      color=get_color(folder_name, freq),
                      linestyle='-',
                      marker='o', s=30,
                      edgecolors='black', linewidth=1)

    else:
        ax[0].scatter(Re, Nu / Re**(1/compensate) , 
                      label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                      color=get_color(folder_name, freq),
                      linestyle='-',
                      marker='o', s=30,
                      edgecolors='black', linewidth=1)
        ax[1].scatter(Ret, Nut / Ret**(1/compensate) , 
                      label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                      color=get_color(folder_name, freq),
                      linestyle='-',
                      marker='o', s=30,
                      edgecolors='black', linewidth=1)
    
if compensate == 0:
    res = np.logspace(3.5,4.3)
    ax[0].plot( res, res*0.4, 'k--', label=r'Nu $\propto$ Re' )  
    res = np.logspace(2.5,4.1)
    ax[1].plot( res, res, 'k--', label=r'Nu $\propto$ Re' )  
    # ax[1].plot( res, res**(1/2) * 30, 'm--', label=r'Nu $\propto$ Re$^{1/2}$' )  
else:
    res = np.logspace(3.5,4.3)
    ax[0].plot( res, res*0.6 / res**(1/compensate), 'k--', label=r'Nu $\propto$ Re' )  
    res = np.logspace(2.5,4.1)
    ax[1].plot( res, res / res**(1/compensate), 'k--', label=r'Nu $\propto$ Re' )  
    # ax[1].plot( res, res**(1/2) * 1.5 / res**(1/compensate), 'm--', label=r'Nu $\propto$ Re$^{1/2}$' )  

fsize = 12
ax[0].tick_params(axis='both', which='both', labelsize=fsize)
ax[1].tick_params(axis='both', which='both', labelsize=fsize)
    
ax[0].set_xscale('log')    
ax[0].set_yscale('log')    
ax[0].set_xlabel(r'$\langle$Re$\rangle$', fontsize=fsize)

    
ax[1].set_xscale('log')    
ax[1].set_yscale('log')    
ax[1].set_xlabel('Re(t)', fontsize=fsize)

if compensate == 0:
    ax[0].set_ylabel(r"$\langle$Nu$\rangle$", fontsize=fsize)
    ax[1].set_ylabel(r'Nu(t)', fontsize=fsize)
elif compensate == 1:
    ax[0].set_ylabel(r"$\langle$Nu$\rangle$ / $\langle$Re$\rangle$", fontsize=fsize)
    ax[1].set_ylabel(r'Nu(t) / Re(t) ', fontsize=fsize)
else:
    ax[0].set_ylabel(rf"$\langle$Nu$\rangle$ / $\langle$Re$\rangle^{{1/{compensate}}}$", fontsize=fsize)
    ax[1].set_ylabel(rf'Nu(t) / Re(t)$^{{1/{compensate}}}$ ', fontsize=fsize)

ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Mass + Frequency", fontsize=fsize)
plt.tight_layout()

if compensate==0:
    filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/Nu_Re.pdf'
else:
    filename = f'/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/Nu_Re_compensated({compensate}).pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()
#%%

umrss = np.array([0.003957, 0.00817, 0.01844, 0.03881, 0.06663])
fr = [1,2,4,8,12]

plt.figure()
plt.plot(fr,umrss,'.-')
# plt.xscale('log')
# plt.yscale('log')
plt.show()


#%%

# =============================================================================
# Solution system
# =============================================================================

plt.rcParams.update({'font.size':13})

def Rs(R,A,B, final_time=1e8):
    if B == 1:
        return (A+1)/(2*R**2) + A*R
   
    elif B > 1:
        pterm = A*R/B
        prod = (A+B) / (6 * np.cbrt(B-1)**2 * np.cbrt(B)**4)
        arcterm = 2*np.sqrt(3) * np.arctan( (1+2*R * np.cbrt(B/(B-1)) ) / np.sqrt(3) )
        logterm = np.log( ( (np.cbrt(B-1) + np.cbrt(B)*R)**2 - np.cbrt(B-1) * np.cbrt(B) * R ) / (np.cbrt(B-1) - np.cbrt(B)*R  )**2  )

        product = np.zeros_like(R)
        product = prod * (arcterm + logterm)
        product[product > final_time] = final_time 
        # product[:-1], product[-1] = prod * (arcterm + logterm), final_time
        return pterm + product

    elif B<1:
        pterm = A*R/B
        prod = (A+B) / (6 * np.cbrt(B-1)**2 * np.cbrt(B)**4)
        arcterm = 2*np.sqrt(3) * np.arctan( (1+2*R * np.cbrt(B/(B-1)) ) / np.sqrt(3) )
        logterm = np.log( ( (np.cbrt(B-1) + np.cbrt(B)*R)**2 - np.cbrt(B-1) * np.cbrt(B) * R ) / (np.cbrt(B-1) - np.cbrt(B)*R )**2  )
        return pterm + prod * (arcterm + logterm)

def T_adim(R,A,B):
    bot = A * (R**3-1) - 1
    top = (B+A) * (R**3-1)        
    
    return 1 - top/bot


numb = 100000
size = 1.5

fig, ax = plt.subplots(1,3, figsize=(15*size,5*size))
# for B in [1.1]:
for B in [0.1,0.5,0.9,0.99]:
    finb = np.cbrt( np.max([0,1-1/B]) )
    R = np.linspace(1,finb,numb, endpoint=True)
    ax[0].plot( Rs(R,1,B) - Rs(R[:1],1,B), R , '-', label='B='+str(B), color=((1-B/2),0,0) ) 
    ax[1].plot( Rs(R,1,B) - Rs(R[:1],1,B), R**3 , '-', label='B='+str(B), color=((1-B/2),0,0) ) 
    ax[2].plot( Rs(R,1,B) - Rs(R[:1],1,B), T_adim(R,1,B) , '-', label='B='+str(B), color=((1-B/2),0,0) ) 

R = np.linspace(1,0,numb) 
ax[0].plot( Rs(R,1,1) - Rs(R[:1],1,1), R, 'k--', label='B=1' )
ax[1].plot( Rs(R,1,1) - Rs(R[:1],1,1), R**3, 'k--', label='B=1' )
ax[2].plot( Rs(R,1,1) - Rs(R[:1],1,1), T_adim(R,1,1) , 'k--', label='B=1' ) 

for B in [1.02,1.2,2]:
    finb = np.cbrt( np.max([0,1-1/B]) )
    R = np.linspace(1,finb,numb, endpoint=True)
    ax[0].plot( Rs(R,1,B) - Rs(R[:1],1,B), R , '-', label='B='+str(B), color=(0,(1-1/B/2),0) ) 
    ax[1].plot( Rs(R,1,B) - Rs(R[:1],1,B), R**3 , '-', label='B='+str(B), color=(0,(1-1/B/2),0) ) 
    ax[2].plot( Rs(R,1,B) - Rs(R[:1],1,B), T_adim(R,1,B) , '-', label='B='+str(B), color=(0,(1-1/B/2),0) ) 
    
for l in range(3):
    ax[l].set_xlabel(r'$Ct$')
    ax[l].legend()
    ax[l].grid()

ax[0].set_xlim(-0.8,50)
ax[0].set_ylabel(r'$\tilde{R}$')
ax[1].set_xlim(-0.2,12)
ax[1].set_ylabel(r'$\tilde{V}$')
ax[2].set_xlim(-0.1,6)
ax[2].set_ylabel(r'$\tilde{T}$')

plt.tight_layout()

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/Solutions.pdf'
# plt.savefig(filename, dpi=400, bbox_inches='tight')

plt.show()

#%%

A = 1
fitfun = lambda Temp,c,A,B: np.real( ( solution_R( R_of_T(Temp, A, B, set_tot_cero=False) , A, B) - solution_R(1, A, B) ) ) / c

plt.figure()

for B in [0.1,0.5,0.9,0.99]:
    finb = np.cbrt( np.max([0,1-1/B]) )
    R = np.linspace(1,finb,numb, endpoint=True)
    # plt.plot( Rs(R,1,B) - Rs(R[:1],1,B), T_adim(R,A,B) , '-', label='B='+str(B), color=((1-B/2),0,0) ) 
    plt.plot( Rs(R,1,B) - Rs(R[:1],1,B), fitfun( T_adim(R,A,B),1,A,B)  , '-', label='B='+str(B), color=((1-B/2),0,0) ) 

R = np.linspace(1,0,numb) 
# plt.plot( Rs(R,1,1) - Rs(R[:1],1,1), T_adim(R,A,1) , 'k--', label='B=1' ) 
plt.plot( Rs(R,1,1) - Rs(R[:1],1,1), fitfun( T_adim(R,A,B),1,A,B) , 'k--', label='B=1' ) 

for B in [1.02,1.2,2]:
    finb = np.cbrt( np.max([0,1-1/B]) )
    R = np.linspace(1,finb,numb, endpoint=True)
    # plt.plot( Rs(R,1,B) - Rs(R[:1],1,B), T_adim(R,A,B) , '-', label='B='+str(B), color=(0,(1-1/B/2),0) ) 
    plt.plot( Rs(R,1,B) - Rs(R[:1],1,B), fitfun( T_adim(R,A,B),1,A,B) , '-', label='B='+str(B), color=(0,(1-1/B/2),0) ) 


plt.xlabel(r'$Ct$')
plt.legend()
plt.grid()

plt.xlim(-0.1,10)
plt.ylim(-0.1,10.1)

plt.ylabel(r'$\tilde{T}$')
plt.show()


#%%
# =============================================================================
# Calibration thermocuples
# =============================================================================

import numpy as np 
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

def to_seconds(timestamps, fmt="%H:%M:%S"):
    t0 = datetime.strptime(str(timestamps[0]), fmt)
    
    times = []
    for t in timestamps:
        strt = str(t)
        if len(strt) > 3: times.append( (datetime.strptime(str(t), fmt)-t0).total_seconds() )
        elif len(strt) == 3: times.append(np.nan)

    return np.array(times)

plot_ladder = 1
show_rampup = 0
deg_fit = [1,2,3]

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Cal thermo/calibration_pt104.csv'
df = pd.read_csv(file_path, delimiter=",", encoding="ISO-8859-1", header=0)
t_pt100, T_pt100 = to_seconds(df['Unnamed: 0'], fmt="%H:%M:%S"), np.array(df['Channel 1 Ave. (C)'])

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Cal thermo/2026-03-13_14.59.48.csv'
df = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1", header=0)
df['Timestamp'] = to_seconds(df["Timestamp"], fmt="%Y-%m-%d_%H.%M.%S.%f")
t_th = np.array(df["Timestamp"][:]) + 48
Tb_th = np.array(df["Water bot °C"][:])
Tt_th = np.array(df["Water top °C"][:])
Tmean_th = ( Tb_th + Tt_th ) / 2

T_pi_int = np.interp(t_th, t_pt100, T_pt100)
end_stair = 308 * 60
fil_pt, fil_th =  t_pt100 < end_stair, t_th < end_stair 


# ----- Initial ladder ramp down ----
if plot_ladder:
    plt.figure()
    # plt.plot( t_pt100 / 60, T_pt100, '.-', label='Pt100' )
    # plt.plot( t_th / 60, Tb_th - 1.1, '.-', label='Bottom thermocuple')
    # plt.plot( t_th / 60, Tt_th, '.-', label='Top thermocuple')
    
    plt.plot( t_pt100[fil_pt] / 60, T_pt100[fil_pt], '.-', label='Pt100' )
    plt.plot( t_th[fil_th] / 60, Tb_th[fil_th], 'g.-', label='Bottom thermocuple')
    plt.plot( t_th[fil_th] / 60, Tt_th[fil_th], 'r.-', label='Top thermocuple')
    plt.plot( t_th[fil_th] / 60, Tmean_th[fil_th], 'y.-', label='Mean thermocuples', markersize=1)
    
    # plt.plot( t_th[fil_th] / 60, Tb_th[fil_th] - Tt_th[fil_th], 'g.-', label='Bottom thermocuple')
    # plt.plot( t_th[fil_th] / 60, Tt_th[fil_th], 'r.-', label='Top thermocuple')
    
    plt.grid()
    plt.legend()
    plt.xlabel('t (min)')
    plt.ylabel('T (°C)')
    plt.show()

fig,ax = plt.subplots(1,len(deg_fit), layout='constrained', figsize=(18,5))
for n in range(len(deg_fit)):
    deg = deg_fit[n]
    #Fit difference
    fit_b = np.polyfit(T_pi_int[fil_th], Tb_th[fil_th] - T_pi_int[fil_th], deg)
    fit_t = np.polyfit(T_pi_int[fil_th], Tt_th[fil_th] - T_pi_int[fil_th], deg)
    fitted_t, fitted_b = np.polyval(fit_t, T_pi_int[fil_th]), np.polyval(fit_b, T_pi_int[fil_th]) 
    
    #Fit conversion
    fit_mean = np.polyfit( Tmean_th[fil_th], T_pi_int[fil_th], deg)
    fitted_mean = np.polyval(fit_mean, Tmean_th[fil_th] )
    
    # plt.plot( T_pi_int[fil_th], Tb_th[fil_th] - T_pi_int[fil_th], 'g.-', label='Bottom', alpha=0.5 )
    # plt.plot( T_pi_int[fil_th], Tt_th[fil_th] - T_pi_int[fil_th], 'r.-', label='Top', alpha=0.5 )
    # plt.plot( T_pi_int[fil_th], fitted_b, 'm--'  )
    # plt.plot( T_pi_int[fil_th], fitted_t, 'y--' )
    
    ax[n].plot( Tmean_th[fil_th], T_pi_int[fil_th], 'r.-', label='Mean', alpha=0.5 )
    ax[n].plot( Tmean_th[fil_th], fitted_mean, '--' )
    
    ax[n].grid()
    ax[n].legend()
    # plt.xlabel(r'$T_{pt}$ (°C)')
    # plt.ylabel(r'$T_{th} - T_{pt}$ (°C)')
    ax[n].set_xlabel(r'$T_{th}$ (°C)')
    ax[n].set_ylabel(r'$T_{pt}$ (°C)')
    ax[n].set_title('Oder fit = '+str(deg))
plt.show()



# ----- Later ramp up ----
if show_rampup:
    plt.figure()
    # plt.plot( t_pt100 / 60, T_pt100, '.-', label='Pt100' )
    # plt.plot( t_th / 60, Tb_th - 1.1, '.-', label='Bottom thermocuple')
    # plt.plot( t_th / 60, Tt_th, '.-', label='Top thermocuple')
    
    plt.plot( t_pt100[~fil_pt] / 60, T_pt100[~fil_pt], '.-', label='Pt100' )
    plt.plot( t_th[~fil_th] / 60, Tb_th[~fil_th], 'g.-', label='Bottom thermocuple')
    plt.plot( t_th[~fil_th] / 60, Tt_th[~fil_th], 'r.-', label='Top thermocuple')
    
    plt.grid()
    plt.legend()
    plt.xlabel('t (min)')
    plt.ylabel('T (°C)')
    plt.show()
    
    plt.figure()
    
    # plt.plot( T_pi_int[fil_th], Tb_th[fil_th], '.-', label='Bottom' )
    # plt.plot( T_pi_int[fil_th], Tt_th[fil_th], '.-', label='Top' )
    
    plt.plot( T_pi_int[fil_th], Tb_th[fil_th] - T_pi_int[fil_th], 'g--', label='Bottom', alpha=0.5 )
    plt.plot( T_pi_int[fil_th], Tt_th[fil_th] - T_pi_int[fil_th], 'r--', label='Top', alpha=0.5 )
    
    plt.plot( T_pi_int[~fil_th], Tb_th[~fil_th] - T_pi_int[~fil_th], 'g.-', label='Bottom' )
    plt.plot( T_pi_int[~fil_th], Tt_th[~fil_th] - T_pi_int[~fil_th], 'r.-', label='Top' )
    
    plt.grid()
    plt.legend()
    plt.xlabel('T pt100 (°C)')
    plt.ylabel('Thermocuple (°C)')
    plt.show()
    
#%%

plt.figure()
# plt.plot( t_pt100 / 60, T_pt100, '.-', label='Pt100' )
# plt.plot( t_th / 60, Tb_th - 1.1, '.-', label='Bottom thermocuple')
# plt.plot( t_th / 60, Tt_th, '.-', label='Top thermocuple')

# plt.plot( t_pt100[fil_pt] / 60, T_pt100[fil_pt], '.-', label='Pt100' )
# # plt.plot( t_th[fil_th] / 60, Tb_th[fil_th], 'g.-', label='Bottom thermocuple')
# # plt.plot( t_th[fil_th] / 60, Tt_th[fil_th], 'r.-', label='Top thermocuple')

# plt.plot( T_pt100, '.-', label='Pt100' )
# plt.plot( Tb_th, 'g.-', label='Bottom thermocuple')
# plt.plot( Tt_th, 'r.-', label='Top thermocuple')

# plt.plot( t_th[fil_th] / 60, Tb_th[fil_th] - fitted_b, 'g--', label='Bottom thermocuple')
# plt.plot( t_th[fil_th] / 60, Tt_th[fil_th] - fitted_t, 'r--', label='Top thermocuple')

start, end = 260*60, 276*60
plt.plot( t_pt100[start:end], T_pt100[start:end], '.-' , label='Pt100'  )
plt.plot( t_th[start*10:end*10], Tb_th[start*10:end*10], '.-' , label='Bottom thermocuple' )
plt.plot( t_th[start*10:end*10], Tt_th[start*10:end*10], '.-' , label='Top thermocuple' )
plt.plot( t_th[start*10:end*10], (Tt_th[start*10:end*10]+Tb_th[start*10:end*10])/2, '.-' , label='Mean thermocuple' )

# plt.hist( T_pt100[start:end], bins=10 )
# plt.hist( Tb_th[start*10:end*10], bins=10 )
# plt.hist( Tt_th[start*10:end*10], bins=10 )
# plt.hist( (Tt_th[start*10:end*10]+Tb_th[start*10:end*10])/2, bins=10)


plt.grid()
plt.legend()
plt.xlabel('t (min)')
plt.ylabel('T (°C)')
plt.show()


print('Pt', np.mean(T_pt100[start:end]), np.std(T_pt100[start:end]) )
print('Bot', np.mean(Tb_th[start*10:end*10]), np.std(Tb_th[start*10:end*10]) )
print('Top', np.mean(Tt_th[start*10:end*10]), np.std(Tt_th[start*10:end*10]) )
print('Both', np.mean( (Tt_th[start*10:end*10]+Tb_th[start*10:end*10])/2 ), np.std( (Tt_th[start*10:end*10]+Tb_th[start*10:end*10])/2 ) )

#%%
# =============================================================================
# Energy loss
# =============================================================================

import numpy as np 
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob


def to_seconds(timestamps, fmt="%Y-%m-%d_%H.%M.%S.%f"):
    t0 = datetime.strptime(str(timestamps[0]), fmt)
    
    times = []
    for t in timestamps:
        strt = str(t)
        if len(strt) > 3: times.append( (datetime.strptime(str(t), fmt)-t0).total_seconds() )
        elif len(strt) == 3: times.append(np.nan)

    return np.array(times)

def to_seconds_ambient(data, limits , fmt1="%Y-%m-%d_%H.%M.%S.%f", fmt2="%Y-%m-%d %H:%M:%S" ):
    t_ini = datetime.strptime(limits[0], fmt1)
    t_end = datetime.strptime(limits[1], fmt1)
    
    times, temps = [], []

    for i in range(len(data)):
        
        t = data['Time'][i]
        T = data['POF_ME201'][i]
        strt = str(t)

        if len(strt) > 3: 
            tdata = datetime.strptime(strt, fmt2)
            
            if (t_ini<tdata) and (tdata<t_end) and ( isinstance(T, str) ):
                times.append( (tdata-t_ini).total_seconds() )    
                temps.append( float(T[:-4]) )
    
    return np.array(times), np.array(temps)

def V_of_T(t, T, A,B,C):
    bot = B + A*T
    argum = 1 - (1-T)/bot 
    loss = C * cumulative_trapezoid((1-T),t, initial=0)

    return  argum - loss/bot

def constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L, Hd, k ):
    alpha = Hd / (rhow*Vo*cp)
    beta = rhoi / rhow
    gamma = Vb/Vo
    kappa = k / (rhow*Vo*cp)
    Ste = L / (cp * (To-Tm))

    A = beta/(gamma+alpha)
    B = A * Ste
    C = kappa /(gamma+alpha)

    return A,B,C, alpha,beta,gamma,kappa,Ste

def density_millero(t, s):
    """
    Computes density of seawater in kg/m^3.
    Function taken from Eq. 6 in Sharqawy2010.
    Valid in the range 0 < t < 40 degC and 0.5 < sal < 43 g/kg.
    Accuracy: 0.01%
    """
    t68 = t / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
    sp = s / 1.00472  # inverse of Eq. 3 in Sharqawy2010

    rho_0 = 999.842594 + 6.793952e-2 * t68 - 9.095290e-3 * t68 ** 2 + 1.001685e-4 * t68 ** 3 - 1.120083e-6 * t68 ** 4 + 6.536336e-9 * t68 ** 5
    A = 8.24493e-1 - 4.0899e-3 * t68 + 7.6438e-5 * t68 ** 2 - 8.2467e-7 * t68 ** 3 + 5.3875e-9 * t68 ** 4
    B = -5.72466e-3 + 1.0227e-4 * t68 - 1.6546e-6 * t68 ** 2
    C = 4.8314e-4
    rho_sw = rho_0 + A * sp + B * sp ** (3 / 2) + C * sp ** 2
    return rho_sw


#------ Thermocuple calibration ------ 
def temperature_calibration( fit_deg ):
    file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Cal thermo/calibration_pt104.csv'
    df = pd.read_csv(file_path, delimiter=",", encoding="ISO-8859-1", header=0)
    t_pt100, T_pt100 = to_seconds(df['Unnamed: 0'], fmt="%H:%M:%S"), np.array(df['Channel 1 Ave. (C)'])
    
    file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Cal thermo/2026-03-13_14.59.48.csv'
    df = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1", header=0)
    df['Timestamp'] = to_seconds(df["Timestamp"], fmt="%Y-%m-%d_%H.%M.%S.%f")
    t_th = np.array(df["Timestamp"][:]) + 48
    Tb_th, Tt_th = np.array(df["Water bot °C"][:]), np.array(df["Water top °C"][:])
    Tmean_th = ( Tb_th + Tt_th ) / 2
    
    T_pi_int = np.interp(t_th, t_pt100, T_pt100)
    end_stair = 308 * 60
    fil_th = t_th < end_stair 

    fit_mean = np.polyfit( Tmean_th[fil_th], T_pi_int[fil_th], deg)
    return fit_mean
    
def correct_temperature( T_mean, coeffs ):
    return np.polyval(coeffs, T_mean )

calibration_deg = 2
coeffs_cal = temperature_calibration( calibration_deg )
correct_temp = lambda T_mean: correct_temperature( T_mean, coeffs_cal )

#%%

rhow, rhoi = 998.2, 916.8 # kg/m3
Tm = 0 #°C
L = 334000 # J/kg 
cp = 4184 # J/(kg K)

Hd = 34320 #500*100 # total heat capacity dodecahedron, J/K
mw = 149 #149 # kg

file_path = '/Volumes/ICESTOCKS/Ice Stocks/Melting/Test1/measures/after-im-test1-4Hz.csv'
df = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1", header=0)

file_temp = '/Volumes/ICESTOCKS/Ice Stocks/ME201-Temperatures/Temperature-data-2026-02-06 15_58_48.csv'
data = pd.read_csv(file_temp, delimiter=",", encoding="ISO-8859-1", header=0)

t_amb, T_amb = to_seconds_ambient(data, list(df["Timestamp"][[0,len(df)-40]]) )
T_ambient = np.nanmean(T_amb)

def sol_expo(t, b,c):
    return T_ambient - (T_ambient-b) * np.exp(-c*t)

print('Initial time: ', datetime.strptime(df["Timestamp"][0], "%Y-%m-%d_%H.%M.%S.%f") )
print('End time: ', datetime.strptime(df["Timestamp"][227599-40], "%Y-%m-%d_%H.%M.%S.%f") )

df['Timestamp'] = to_seconds(df["Timestamp"])
 
t_cal = np.array(df["Timestamp"][620:])
Tb_w_cal = np.array(df["Water bot °C"][620:])
Tt_w_cal = np.array(df["Water top °C"][620:])

T_w_cal = correct_temp( (Tt_w_cal+Tb_w_cal)/2 )

fil = np.isnan(t_cal)
(b2,c2), _ = curve_fit(sol_expo, t_cal[~fil], T_w_cal[~fil], p0=(15,0) )
print(f'T_ini = {b2:.2f} °C')
print(f'C = {c2:.4} 1/s')
print()
k = c2 * (mw*cp + Hd)
print(f'k = {k:.4} J/sK')
# C = 2.018e-05 1/s
#k = 13.27 o  J/sK

plt.figure()
plt.plot( t_cal, T_w_cal,'b-', label=r'$T_{bath}$' )
plt.plot( t_amb, T_amb, 'g.-', label=r'$T_{amb}$' )

plt.plot( t_cal, sol_expo(t_cal,b2,c2), 'r--', label='fit' )

plt.legend()
plt.xlabel('time (s)')
plt.ylabel('Temperature (°C)')

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/fit.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()

#%%

# new heat loss coefficient
cp = 4184 # J/(kg K)
mw = 122 # kg
Hd = 34320 # total heat capacity dodecahedron, J/K


n = 1

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps/'
mes = '*heat-abs*'

files = glob.glob( file_path + mes )


df = pd.read_csv(files[n], delimiter=";", encoding="ISO-8859-1", header=0)

file_temp = '/Volumes/ICESTOCKS/Ice Stocks/ME201-Temperatures/Temperature-data-2026-02-23 16_22_02.csv'
data = pd.read_csv(file_temp, delimiter=",", encoding="ISO-8859-1", header=0)

t_amb, T_amb = to_seconds_ambient(data, list(df["Timestamp"][[0,len(df)-100]]) )
T_ambient = np.nanmean(T_amb)

def sol_expo(t, b,c):
    return T_ambient - (T_ambient-b) * np.exp(-c*t)

print('Initial time: ', datetime.strptime(df["Timestamp"][0], "%Y-%m-%d_%H.%M.%S.%f") )
print('End time: ', datetime.strptime(df["Timestamp"][len(df) - 100], "%Y-%m-%d_%H.%M.%S.%f") )

df['Timestamp'] = to_seconds(df["Timestamp"])
 
t_cal = np.array(df["Timestamp"][:])
# T_w_cal = np.array(df["Water top °C"][:])

Tb_w_cal = np.array(df["Water bot °C"][:])
Tt_w_cal = np.array(df["Water top °C"][:])

T_w_cal = correct_temp( (Tt_w_cal+Tb_w_cal)/2 )


fil = np.isnan(t_cal)
(b2,c2), cov = curve_fit(sol_expo, t_cal[~fil], T_w_cal[~fil], p0=(15,0) )
print(f'T_ini = {b2:.2f} °C')
print(f'C = {c2:.4} 1/s')
print(f'Covariace matrix = {cov}')
print()
k = c2 * (mw*cp + Hd)
print(f'k = {k:.4} J/sK')

# n = 0,  C = 2.323e-05 1/s,  k = 11.68 J/sK
# n = 1,  C = 2.332e-05 1/s,  k = 11.73 J/sK

# With corrected temperature
# n = 0,  C = 2.128e-05 1/s,  k = 11.59 J/sK
# n = 1,  C = 2.112e-05 1/s,  k = 11.51 J/sK


plt.figure()
plt.plot( t_cal, T_w_cal,'b-', label=r'$T_{bath}$' )
plt.plot( t_amb, T_amb, 'g.-', label=r'$T_{amb}$' )

plt.plot( t_cal, sol_expo(t_cal,b2,c2), 'r--', label='fit' )
# tall = np.linspace(0,1,10000) * 60**2 * 15
# plt.plot( tall, sol_expo(tall,b2,c2), 'r--', label='fit' )

plt.legend()
plt.xlabel('time (s)')
plt.ylabel('Temperature (°C)')

plt.show()



#%%
# =============================================================================
# Energy balance
# =============================================================================

import numpy as np 
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
from scipy.integrate import cumulative_trapezoid

# from matplotlib import cm
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, least_squares
# from scipy.stats import linregress

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

def bt_nm(lam,t,T, tdiff):
    s = 2/lam
    beta = tdiff * lam**2
    ct = exp_convolution(t, T, beta)
    t1 = np.exp(-beta*t) 
    return s/(lam) * ( t1 - T + beta * ct )

def energy_delayed(t,T, tdiff, N=1000):

    lam = np.pi/2 * (1 + 2* np.arange(0,N)) 
    mb = np.zeros_like(t)
    for n in range(0,N):        
        mbn = bt_nm(lam[n], t, T, tdiff) 
        
        mb += mbn
        
    return mb


def to_seconds(timestamps, fmt="%Y-%m-%d_%H.%M.%S.%f"):
    t0 = datetime.strptime(str(timestamps[0]), fmt)
    
    times = []
    for t in timestamps:
        strt = str(t)
        if len(strt) > 3: times.append( (datetime.strptime(str(t), fmt)-t0).total_seconds() )
        elif len(strt) == 3: times.append(np.nan)

    return np.array(times)

def constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L, Hd, k ):
    alpha = Hd / (rhow*Vo*cp)
    beta = rhoi / rhow
    gamma = Vb/Vo
    kappa = k / (rhow*Vo*cp)
    Ste = L / (cp * (To-Tm))

    A = beta/(gamma+alpha)
    B = A * Ste
    C = kappa /(gamma+alpha)
    M = alpha / (gamma+alpha)

    return A,B,C,M, alpha,beta,gamma,kappa,Ste    


def rk4_sol(dt, A,B,C,D):
    
    def func(t,R):
        top = B * (R**3 -1) + 1 + C/D * (R+t-1)
        bot = A * (R**3 -1) - 1
        return top/bot
    
    R_sol = [1]
    t = [0]

    # for n in range(1,len(t)):
    while R_sol[-1] > 0:
        
        rn, tn = R_sol[-1], t[-1]
        k1 = func(tn, rn)
        k2 = func(tn + dt/2, rn + k1 * dt/2)
        k3 = func(tn + dt/2, rn + k2 * dt/2)
        k4 = func(tn + dt, rn + k3 * dt)
        
        rnext = rn + dt/6 * (k1+2*k2+2*k3+k4)
        
        R_sol.append( rnext )
        t.append( tn + dt )    
    
    return np.array(R_sol), np.array(t)

def solution_R( R, A,B ):
    pterm = A*R/B
    prod = (A+B) / (6 * np.cbrt(B-1)**2 * np.cbrt(B)**4)
    arcterm = 2*np.sqrt(3) * np.arctan( (1+2*R * np.cbrt(B/(B-1)) ) / np.sqrt(3) )
    logterm = np.log( ( (np.cbrt(B-1) + np.cbrt(B)*R)**2 - np.cbrt(B-1) * np.cbrt(B) * R ) / (np.cbrt(B-1) - np.cbrt(B)*R )**2  )
    return pterm + prod * (arcterm + logterm)


def V_of_T(t,T, A,B,C):
    t1 = 1 - (1-T)/(B+A*T)
    inte = cumulative_trapezoid(1-T, t, initial=0)
    return t1 - C/(B+A*T) * inte

def V_of_T_d(t,T, A,B,C, M, tdiff, N=1000):
    t1 = 1 - (1-T)/(B+A*T)
    t2 = M * energy_delayed(t, T, tdiff, N=N) /(B+A*T)
    inte = cumulative_trapezoid(1-T, t, initial=0)
    return t1 + t2 - C/(B+A*T) * inte

def energies(t,T, alpha,beta,gamma,kappa,Ste):
    eb = 1-T
    ed = alpha/gamma * (1-T)
    el = kappa/gamma * cumulative_trapezoid(1-T, t, initial=0)

    A = beta/(gamma+alpha)
    B = A * Ste
    C = kappa /(gamma+alpha)
    V = V_of_T(t, T, A, B, C)
    
    elat = beta/gamma * Ste * (1-V)
    emelt = beta/gamma (1-V) * T

    return eb,ed,el, elat, emelt
    
def get_color(mass, freq):
    folder_colors = {'5': cm.Blues, '10': cm.Greens, '20': cm.Reds, '15': cm.Greys }
    frequencies = ['1','2','4','8','12']

    cmap = folder_colors[mass]
    freq_index = frequencies.index(freq)
    return cmap(0.3 + 0.7 * freq_index / (len(frequencies) - 1))  # soft to dark

    
def get_temp( name, start, temp_correction=1 ):
    df = pd.read_csv(name, delimiter=";", encoding="ISO-8859-1", header=0)    
    df['Timestamp'] = to_seconds(df["Timestamp"])

    t = np.array(df["Timestamp"][:])
    Tt = np.array(df["Water top °C"][:])
    Tb = np.array(df["Water bot °C"][:])

    if temp_correction:
        Tmean = correct_temp( (Tt + Tb) / 2 )
        t, T = t[start:] - t[start], Tmean[start:] 
        T0 = T[0]

    else:
        T0 = Tb[0] # °C # maybe should use Tt
        T = T0 + (Tt-Tt[0] + Tb-Tb[0] ) / 2
        t, T = t[start:] - t[start], T[start:] 
        
    return t, T



def plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, temp_correction=1, delayed=0, Hd_fit=0, delimiter=";", encoding="ISO-8859-1", header=0):
    df = pd.read_csv(name, delimiter=";", encoding="ISO-8859-1", header=0)    
    df['Timestamp'] = to_seconds(df["Timestamp"])

    t = np.array(df["Timestamp"][:])
    Tt = np.array(df["Water top °C"][:])
    Tb = np.array(df["Water bot °C"][:])
        
    if temp_correction:
        Tmean = correct_temp( (Tt + Tb) / 2 )
        t, T = t[start:] - t[start], Tmean[start:] 
        T0 = T[0]
        # plt.title('With tempretature correction')
    else:
        T0 = Tb[0] # °C # maybe should use Tt
        T = T0 + (Tt-Tt[0] + Tb-Tb[0] ) / 2
        t, T = t[start:] - t[start], T[start:] 
        plt.title('Without tempretature correction')
        
    
    Vb = (bathmass )/ rhow # m^3
    V0 = 1.0 * icemass / rhoi # m^3

    # Vb = (mass_bath[mass] - 10.)/ rhow # m^3
    # V0 = 1.0 * float(mass) / rhoi # m^3

    T_tilde = (T - Tm) / (T0-Tm)

    if Hd_fit:
        def fit_hd(val): #Using minimum ±0.2 min
            A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0 * 1.0, Vb, rhoi, rhow, cp, L, val[0], k )    
            V_tilde = V_of_T(t, T_tilde, A, B, C)
            minar = np.nanargmin(V_tilde)
            minmean = V_tilde[minar-120:minar+120]  #np.nanmean( V_tilde[minar-120:minar+120] )
            return  minmean[~np.isnan(minmean)] 
        ls = least_squares(fit_hd, [Hd], bounds=((0.),(Hd*1.5)), method='trf')
        # ls = least_squares(fit_hd, [Hd], method='lm')

        A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0 * 1.0, Vb, rhoi, rhow, cp, L, ls.x[0], k )
        # print( ls.x )

    else:
        A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0 * 1.0, Vb, rhoi, rhow, cp, L, Hd, k )
        
    ctes = [A,B,C,M, alpha,beta,gamma,kappa,Ste]
     
    if delayed:
        V_tilde = V_of_T_d(t, T_tilde, A, B, C, M, tdiff_st, N=70)
        m = (1 - V_tilde) * V0 * rhoi
    else:
        V_tilde = V_of_T(t, T_tilde, A, B, C)
        m = (1 - V_tilde) * V0 * rhoi
    
    # print(np.nanmin(V_tilde), t[np.nanargmin(V_tilde)])
    # print(np.nanmin(T), end=', ')
    # print(t[np.nanargmin(V_tilde)], end=', ')
    
    if ploty == 'T':
        plt.plot(t/60, T, '.-', color=color, label=f'{mass} kg, {freq} Hz')
    elif ploty == 'Tt':
        plt.plot(t/60, T_tilde, '.-', color=color, label=f'{mass} kg, {freq} Hz')
    elif ploty == 'Vt':
        plt.plot(t/60  , V_tilde, '.-', color=color,  label=f'{mass} kg, {freq} Hz')
    elif ploty == 'm':
        plt.plot(t/60, m , '.-', color=color,  label=f'{mass} kg, {freq} Hz')

    return t, T, T_tilde, V_tilde, m, ctes


#------ Thermocuple calibration ------ 
def temperature_calibration( fit_deg ):
    file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Cal thermo/calibration_pt104.csv'
    df = pd.read_csv(file_path, delimiter=",", encoding="ISO-8859-1", header=0)
    t_pt100, T_pt100 = to_seconds(df['Unnamed: 0'], fmt="%H:%M:%S"), np.array(df['Channel 1 Ave. (C)'])
    
    file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Cal thermo/2026-03-13_14.59.48.csv'
    df = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1", header=0)
    df['Timestamp'] = to_seconds(df["Timestamp"], fmt="%Y-%m-%d_%H.%M.%S.%f")
    t_th = np.array(df["Timestamp"][:]) + 48
    Tb_th, Tt_th = np.array(df["Water bot °C"][:]), np.array(df["Water top °C"][:])
    Tmean_th = ( Tb_th + Tt_th ) / 2
    
    T_pi_int = np.interp(t_th, t_pt100, T_pt100)
    end_stair = 308 * 60
    fil_th = t_th < end_stair 

    fit_mean = np.polyfit( Tmean_th[fil_th], T_pi_int[fil_th], fit_deg)
    return fit_mean
    
def correct_temperature( T_mean, coeffs ):
    return np.polyval(coeffs, T_mean )


#------ Definitions and values ------ 
starts = {'20,4':198, '20,2':139, '10,12':140, '10,8':95, '10,4':125, '10,2':240, '10,1':300, '5,12':165, '5,8':90, '5,4':130,
          '5,2':170, '5,1':120, '20,1':160, '20,12':164, '20,8':118, '15,4':0 }

mass_ice = {'5,1':4.992, '5,2':4.997, '5,4':5.009, '5,8':5.011, '5,12':5.024, '10,1':4.977+5.015, '10,2':5.037+5.006, '10,4':5.011+5.036, 
            '10,8':5.013+5.024, '10,12':4.997+5.008, '20,1':5.015+4.970+4.965+4.949, '20,2':5.015+4.970+4.965+4.949, '20,4':5.016+5.015+5.001+4.987, 
            '20,8':5.052+5.021+5.013+4.931, '20,12':5.011+5.023+5.010+5.010}

mass_bath = {'10':112, '5':117, '20':102, '15':107}

rhow, rhoi = 998.2, 916.8 # kg/m3
Tm = 0 #°C
L = 334000 # J/kg 
cp = 4184 # J/(kg K)

k = 11.59 # J/sK
Hd = 34320 #J/K
Tm = 0 #°C

calibration_deg = 2
coeffs_cal = temperature_calibration( calibration_deg )
correct_temp = lambda T_mean: correct_temperature( T_mean, coeffs_cal )

#%%

To, Tm = 20,  0
Vb = 102 / rhow
Vo = 10 / rhoi


A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L, Hd, k )


D = 0.001
dt = 0.01

rsol, t = rk4_sol(dt, A, B, C, D)    
rsol0, t0 = rk4_sol(dt, A, B, 0, D)    


rana = np.linspace(0,1,100) 
tana = solution_R( rana, A, B) - solution_R( 1, A, B)

plt.figure()

plt.plot(t / D, rsol**3, label='rk4')
plt.plot(t0 / D, rsol0**3, label='rk4')

plt.plot(tana / D, rana**3, '.', label='Analytic')

plt.grid()
plt.legend()
plt.show()



#%%
# latest data

# tdiff_st = 0.05 # 1/s (diffusivity time of 8.5 mm of steel , alpha/L^2 = (4/1e6) / (0.0085)**2 )
tdiff_st = 1e-4 # 1/s (trying values )

delayed = 0
temp_correction = 1
hdfit = 1
ploty = 'Vt'

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps/'

names = '*Hz.csv'

file_name = glob.glob(file_path+names)
# file_name = file_name[:13]+file_name[14:]

avoid_mass = ['15'] #'10','15','20']
avoid_freq = []

plt.figure()

for name in file_name:
    splits = name[-20:].split('-')[-2:]     
    mass, freq = splits[0][:-2], splits[1][:-6]

    if (mass not in avoid_mass) and (freq not in avoid_freq):
        start = starts[mass+','+freq]
        color = get_color(mass, freq)
        
        bathmass = mass_bath[mass] - 10.
        if mass+','+freq != '15,4':
            icemass = mass_ice[mass+','+freq]
        else: icemass = float(mass)
        
        # print(mass, freq, end= ' ')
        
        t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, temp_correction=temp_correction, delayed=delayed, Hd_fit=hdfit )
        # t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, 9108, k, ploty, start, color, temp_correction=temp_correction, delayed=delayed, Hd_fit=hdfit )
        
        # plt.plot(t/60, T, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, T_tilde, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60  , V_tilde, '.-', color=color,  label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, m , '.-', color=color,  label=f'{mass} kg, {freq} Hz')

    

plt.xlabel(r'$t$ (min)')
# plt.ylabel(r'$T$ (°C)')
plt.ylabel(r'$\tilde{V}$')
# plt.ylabel(ploty)

# plt.title(r'Without $Hd$ fit')
# plt.title(r'With $Hd$ fit')

plt.ylim(-0.12,1.05)

plt.legend()
plt.grid()

savepath = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/'
savename = '5kg_exp.png'
# plt.savefig(savepath+savename, dpi=100, bbox_inches='tight')


plt.show()

#%%



def Rs(R,A,B, final_time=1e8):
    if B == 1:
        return (A+1)/(2*R**2) + A*R
   
    elif B > 1:
        pterm = A*R/B
        prod = (A+B) / (6 * np.cbrt(B-1)**2 * np.cbrt(B)**4)
        arcterm = 2*np.sqrt(3) * np.arctan( (1+2*R * np.cbrt(B/(B-1)) ) / np.sqrt(3) )
        logterm = np.log( ( (np.cbrt(B-1) + np.cbrt(B)*R)**2 - np.cbrt(B-1) * np.cbrt(B) * R ) / (np.cbrt(B-1) - np.cbrt(B)*R  )**2  )

        product = np.zeros_like(R)
        product = prod * (arcterm + logterm)
        product[product > final_time] = final_time 
        # product[:-1], product[-1] = prod * (arcterm + logterm), final_time
        return pterm + product

    elif B<1:
        pterm = A*R/B
        prod = (A+B) / (6 * np.cbrt(B-1)**2 * np.cbrt(B)**4)
        arcterm = 2*np.sqrt(3) * np.arctan( (1+2*R * np.cbrt(B/(B-1)) ) / np.sqrt(3) )
        logterm = np.log( ( (np.cbrt(B-1) + np.cbrt(B)*R)**2 - np.cbrt(B-1) * np.cbrt(B) * R ) / (np.cbrt(B-1) - np.cbrt(B)*R )**2  )
        return pterm + prod * (arcterm + logterm)

def T_adim(R,A,B):
    bot = A * (R**3-1) - 1
    top = (B+A) * (R**3-1)        
    
    return 1 - top/bot

def Rt(T,A,B):
    bot = 1-T
    top = (B+A*T)
    
    return np.cbrt(1 - top/bot )


def solution_R( R, A,B ):
    pterm = A*R/B
    prod = (A+B) / (6 * np.cbrt(B-1)**2 * np.cbrt(B)**4)
    arcterm = 2*np.sqrt(3) * np.arctan( (1+2*R * np.cbrt(B/(B-1)) ) / np.sqrt(3) )
    logterm = np.log( ( (np.cbrt(B-1) + np.cbrt(B)*R)**2 - np.cbrt(B-1) * np.cbrt(B) * R ) / (np.cbrt(B-1) - np.cbrt(B)*R )**2  )
    return pterm + prod * (arcterm + logterm)

# def solution_T( R, A,B,Tm,To ):
#     u = 1 - R**3
#     top = (Tm*A + B*(To-Tm)) * u + To
#     bot = A*u + 1
#     return top/bot

def R_of_T(T, A,B, set_tot_cero=False):
    bot =  B + A*T
    argum = 1 - (1-T)/bot 
    if set_tot_cero:
        argum[argum<0] = 0
    return  np.cbrt(argum)

ploty = None

avoid_mass = ['15','20','5'] #'10','15','20']
avoid_freq = []

plt.figure()

for name in file_name:
    splits = name[-20:].split('-')[-2:]     
    mass, freq = splits[0][:-2], splits[1][:-6]

    if (mass not in avoid_mass) and (freq not in avoid_freq):
        start = starts[mass+','+freq]
        color = get_color(mass, freq)
        
        bathmass = mass_bath[mass] - 10.
        if mass+','+freq != '15,4':
            icemass = mass_ice[mass+','+freq]
        else: icemass = float(mass)
        
        # print(mass, freq, end= ' ')
        
        t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, 
                                                      temp_correction=temp_correction, delayed=delayed, Hd_fit=hdfit )
        A,B,C,M, alpha,beta,gamma,kappa,Ste = ctes
        
        sokl = np.real( ( solution_R( R_of_T(T_tilde, A, B, set_tot_cero=False) , A, B) - solution_R(1, A, B) ) )
        
        plt.plot(t/60, sokl , '.-', color=color, label=f'{mass} kg, {freq} Hz')


plt.xlabel(r'$t$ (min)')
plt.ylabel(r'$R_S(R_T(T)) - R_S(1)$')

savepath = './Documents/'
savename = '10kg_sol.png'
plt.savefig(savepath+savename, dpi=100, bbox_inches='tight')

plt.show()


#%%

hds = np.array( [51479.99997672, 51479.98898221, 17289.79140089, 21898.5924872,36085.21329777,29233.6264102,20483.22514788,13828.85406739,
       1653.36897302,11833.03504191,12576.65410334,6046.59082247,46261.20398587,42430.36947115,37876.73435438,35731.69642151] )

temps = np.array( [4.069517088948605,4.531061131295094,11.355957488441915,11.2517734918756,11.408058967103017,11.408058967103017,11.355957488441915,15.385208433859384,
         15.280104086294719,15.490335621835714,15.385208433859384,15.437769304250535,5.3527784517113135,3.403696484718621,3.3525213254088095,7.568720880875728] )

times = np.array( [1602.694, 3455.982, 144.398, 282.2, 475.994, 835.796, 548.5989999999999, 98.402, 131.399, 317.299, 400.297, 398.296, 7827.964, 270.797, 591.495, 990.595 ])


savepath = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/'
savename = 'fithd.png'
# plt.savefig(savepath+savename, dpi=100, bbox_inches='tight')

plt.figure()
# plt.plot( times, temps, '.')
plt.plot( times/60, hds, '.')
plt.xlabel('t (min)')
plt.ylabel('Hd')

savename = 'hd_time.pdf'
# plt.savefig(savepath+savename, dpi=100, bbox_inches='tight')
plt.show()

plt.figure()
# plt.plot( times, temps, '.')
plt.plot( temps, hds, '.')
plt.xlabel('Temp min (°C)')
plt.ylabel('Hd')

savename = 'hd_temp.pdf'
# plt.savefig(savepath+savename, dpi=100, bbox_inches='tight')
plt.show()

plt.figure()
ss = plt.scatter(times, temps, c=hds, cmap='jet')
plt.xlabel('t (min)')
plt.ylabel('Temp min (°C)')
plt.colorbar(ss, label='Hd')

savename = 'hd_scatter.pdf'
# plt.savefig(savepath+savename, dpi=100, bbox_inches='tight')
plt.show()

#%%
# repetitions data

delayed = 0
temp_correction = 1
hdfit = 0
ploty = None # 'T', 'Tt', 'Vt', or 'm'

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Repetitions/'
names = '*Hz.csv'
file_name = glob.glob(file_path+names)

avoid_mass = [] # ['10','15','20']
avoid_freq = [] # ['1','2']
avoid_repe = [] # ['1','2']

rep_starts = {'1,5,4':120, '1,5,8':132, '1,5,12':132, '2,5,4':269, '2,5,8':180, '2,5,12':120, '3,5,12':85, '3,5,8':162, '3,5,4':207,
              '4,5,8':382, '2,10,4':260, '4,5,4':213}

rep_ice = {'1,5,4':4.874, '1,5,8':4.924, '1,5,12':4.907, '2,5,4':5.003, '2,5,8':5.008, '2,5,12':4.980, '3,5,12':4.829, '3,5,8':4.997,
           '3,5,4':4.995, '4,5,8':4.922, '2,10,4':10.125, '4,5,4':4.964}

plt.figure()

for name in file_name:
    splits = name[-30:].split('-')[-3:]     
    repe, mass, freq = splits[0][-1], splits[1][:-2], splits[2][:-6]

    if (mass not in avoid_mass) and (freq not in avoid_freq) and (repe not in avoid_repe):
        start = rep_starts[repe+','+mass+','+freq]
        # color = get_color('15', freq)
        color = get_color(mass, freq)
        
        bathmass = mass_bath[mass] - 10.
        icemass = rep_ice[repe+','+mass+','+freq]
        
        
        t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, temp_correction=temp_correction, delayed=delayed, Hd_fit=hdfit  )
        
        vv = 2
        if mass=='10': vv = 1
        
        # plt.plot(t/60, T, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, T_tilde, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        plt.plot(t/60  , V_tilde, '-', color=color,  label=f'Rep {int(repe)-vv}, {mass} kg, {freq} Hz')
        # plt.plot(t/60, m , '.-', color=color,  label=f'{mass} kg, {freq} Hz')
        

show = ['5,12', '5,8', '5,4', '10,4']        

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps/'
names = '*Hz.csv'
file_name = glob.glob(file_path+names)

for name in file_name:
    splits = name[-20:].split('-')[-2:]     
    mass, freq = splits[0][:-2], splits[1][:-6]

    # if (mass not in avoid_mass) and (freq not in avoid_freq):
    if mass+','+freq in show:
        start = starts[mass+','+freq]
        # print(start)
        # start = 0
        color = get_color(mass, freq)
        
        bathmass = mass_bath[mass] - 10.
        if mass+','+freq != '15,4':
            icemass = mass_ice[mass+','+freq]
        else: icemass = float(mass)
        
        t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, temp_correction=temp_correction, delayed=delayed )
        
        # plt.plot(t/60, T, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, T_tilde, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        plt.plot(t/60  , V_tilde, '.-', color=color,  label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, m , '.-', color=color,  label=f'{mass} kg, {freq} Hz')

    

plt.xlabel(r'$t$ (min)')
# plt.ylabel(r'$T$ (°C)')
plt.ylabel(r'$\tilde{V}$')
# plt.ylabel(ploty)

plt.legend()
plt.grid()

savepath = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/MMMM-april26/'
savename = 'repetitions.png'
plt.savefig(savepath+savename, dpi=100, bbox_inches='tight')

plt.show()

#%%
# dif sizes

delayed = 0
temp_correction = 1
hdfit = 1
ploty = None # 'T', 'Tt', 'Vt', or 'm'

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps-10mm/'
names = '*Hz.csv'
file_name = glob.glob(file_path+names)

avoid_mass = ['5','15','20']
avoid_freq = ['2','8','4']
avoid_repe = []

rep_starts = {'0,10,4':99, '1,10,4':246, '0,10,12':201, '0,10,1':414 }
rep_ice = {'0,10,4':9.471+0.607, '1,10,4':10.065, '0,10,12':9.203, '0,10,1':9.992 }

plt.figure()

for name in file_name:
    splits = name[-30:].split('-')[-4:]     
    repe, mass, freq = splits[0][-1], splits[2][:-2], splits[3][:-6]

    if (mass not in avoid_mass) and (freq not in avoid_freq) and (repe not in avoid_repe):
        start = rep_starts[repe+','+mass+','+freq]
        color = get_color('15', freq)
        # color = get_color(mass, freq)
        
        bathmass = mass_bath[mass] - 10.
        icemass = rep_ice[repe+','+mass+','+freq]
        
        print(bathmass, icemass)
        
        t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, 
                                                      temp_correction=temp_correction, delayed=delayed, Hd_fit=hdfit  )
        
        # plt.plot(t/60, T, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, T_tilde, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        plt.plot(t/60  , V_tilde, '.-', color=color,  label=f'10 mm, {freq} Hz')
        # plt.plot(t/60, m , '.-', color=color,  label=f'{mass} kg, {freq} Hz')
        
        
        
file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps-50mm/'
names = '*Hz.csv'
file_name = glob.glob(file_path+names)

avoid_mass = ['5','15','20']
avoid_freq = ['2','8','4']
avoid_repe = ['0']

rep_starts = {'1,10,4':444, '2,10,4':330 }
rep_ice = {'1,10,4':9.246, '2,10,4':10.312 }


for name in file_name:
    splits = name[-30:].split('-')[-4:]     
    repe, mass, freq = splits[0][-1], splits[2][:-2], splits[3][:-6]

    if (mass not in avoid_mass) and (freq not in avoid_freq) and (repe not in avoid_repe):
        start = rep_starts[repe+','+mass+','+freq]
        color = get_color('20', freq)
        # color = get_color(mass, freq)
        
        bathmass = mass_bath[mass] - 10.
        if repe+','+mass+','+freq == '1,10,4':
            bathmass = mass_bath[mass] - 5.
        icemass = rep_ice[repe+','+mass+','+freq]
        
        print(bathmass, icemass)
        
        t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, 
                                                      temp_correction=temp_correction, delayed=delayed, Hd_fit=hdfit  )
        
        # plt.plot(t/60, T, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, T_tilde, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        plt.plot(t/60  , V_tilde, '.-', color=color,  label=f'50 mm, {freq} Hz')
        # plt.plot(t/60, m , '.-', color=color,  label=f'{mass} kg, {freq} Hz')
        
        

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps/'
names = '*Hz.csv'
file_name = glob.glob(file_path+names)

for name in file_name:
    splits = name[-20:].split('-')[-2:]     
    mass, freq = splits[0][:-2], splits[1][:-6]

    if (mass not in avoid_mass) and (freq not in avoid_freq):
        start = starts[mass+','+freq]
        # print(start)
        # start = 0
        color = get_color(mass, freq)
        
        bathmass = mass_bath[mass] - 10.
        if mass+','+freq != '15,4':
            icemass = mass_ice[mass+','+freq]
        else: icemass = float(mass)
        
        print(bathmass, icemass)
        
        t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, 
                                                      temp_correction=temp_correction, delayed=delayed, Hd_fit=hdfit  )
        
        # plt.plot(t/60, T, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, T_tilde, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        plt.plot(t/60  , V_tilde, '.-', color=color,  label=f'25 mm, {freq} Hz')
        # plt.plot(t/60, m , '.-', color=color,  label=f'{mass} kg, {freq} Hz')

    

plt.xlabel(r'$t$ (min)')
# plt.ylabel(r'$T$ (°C)')
# plt.ylabel(r'$\tilde{V}$')
# plt.ylabel(ploty)
plt.ylabel(r'$\tilde{V}$')

plt.legend()
plt.grid()

savepath = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/MMMM-april26/'
savename = 'sizes(other).png'
# plt.savefig(savepath+savename, dpi=100, bbox_inches='tight')

plt.show()



#%%
# diff_st, L_st = 0.05 /1e6, 0.0085 # m^2/s , m (diffusivity of steel, and thickness of wall)
tdiff_st = 0.0553 # 1/s (trying values )

plt.figure()

# for name in file_name[0]:
for name in [15]:
    splits = file_name[name][-20:].split('-')[-2:]     
    mass, freq = splits[0][:-2], splits[1][:-6]
    
    print(f'{mass} kg, {freq} Hz')

    df = pd.read_csv(file_name[name], delimiter=";", encoding="ISO-8859-1", header=0)    
    df['Timestamp'] = to_seconds(df["Timestamp"])

    t = np.array(df["Timestamp"][:])

    Tt = np.array(df["Water top °C"][:])
    Tb = np.array(df["Water bot °C"][:])
    
    T0 = Tb[0]
    T = T0 + (Tt-Tt[0] + Tb-Tb[0] ) / 2
    
    t, T = t[start:] - t[start], T[start:] 
    
    Vb = (mass_bath[mass] - 0.)/ rhow # m^3
    V0 = 1.0 * float(mass) / rhoi # m^3

    A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0 * 0.9, Vb, rhoi, rhow, cp, L, Hd, k )    
    # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, 0, k )    
    # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, 0 )    
    
    T_tilde = (T - Tm) / (T0-Tm)

    V_tilde = V_of_T(t, T_tilde, A, B, C)
    V_tilde_d = V_of_T_d(t, T_tilde, A, B, C, M, tdiff_st, N=50)


    # V_tilde_d100 = V_of_T_d(t, T_tilde, A, B, C, M, diff_st, L_st, N=50)
    # V_tilde_d1000 = V_of_T_d(t, T_tilde, A, B, C, M, diff_st, L_st, N=1000)

    # m = (1 - V_tilde) * V0 * rhoi

    
    # plt.plot( Tt - Tt[0], '.-')
    # plt.plot( Tb - Tb[0], '.-')
    # plt.plot( T - T[0], '.-')
    
    # plt.plot(t/60, V_tilde, '.-', label=r'$\tilde{V}$')
    # plt.plot(t/60, V_tilde_d, '.-', label=r'$\tilde{V}_{delayed}$')
    plt.plot( V_tilde, '.-', label=r'$\tilde{V}$')
    plt.plot( V_tilde_d, '.-', label=r'$\tilde{V}_{delayed}$')

    # plt.plot(t/60, V_tilde_d100 / V_tilde_d1000 , '.-', label=r'$\tilde{V}_{delayed}$')
    
    plt.title( f'{mass} kg, {freq} Hz' )

plt.xlabel(r'$t$ (min)')
plt.ylabel(r'$T$ (°C)')
plt.grid()
plt.legend()
plt.show()


#%%
# old data

k_old = 13.7 # J/sK
tdiff_st = 0.05 # 1/s (diffusivity time of 8.5 mm of steel , alpha/L^2 = (4/1e6) / (0.0085)**2 )
# tdiff_st = 1e-4 # 1/s (trying values )
delayed = 0

folder_paths = ['/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Test7-5kg-experiment/Temperature Recordings/',
                '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Test5-10kg-experiment/Temperature Recordings/',
                '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Test6-20kg/Temperature Recordings/' ]
indexes = {0:[1,-2,-4,-2], 1:[1,-2,-3,-2], 2:[-1,-10,-4,-2] }

names = '*.csv'

plt.figure()

for j in range(len(folder_paths)):
    file_name = glob.glob(folder_paths[j]+names)
    indx = indexes[j]
    for n in range(len(file_name)):

        splits = file_name[n][:].split('-')
        mass, freq = splits[indx[0]][:indx[1]], splits[indx[2]][:indx[3]]
        
        color = get_color(mass, freq)
    
        df = pd.read_csv(file_name[n], delimiter=";", encoding="ISO-8859-1", header=0)    
        df['Timestamp'] = to_seconds(df["Timestamp"])
    
        t = np.array(df["Timestamp"][:])
        Tt = np.array(df["Water top °C"][:])
        Tb = np.array(df["Water bot °C"][:])
        
        T0 = Tb[0] # °C
        T = T0 + (Tt-Tt[0] + Tb-Tb[0] ) / 2
        t, T = t[start:] - t[start], T[start:] 
        
        Vb = (mass_bath[mass] - 0)/ rhow # m^3
        V0 = 0.97 * float(mass) / rhoi # m^3
    
        A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, k_old )    
        # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, 0, k )    
        # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, 0 )    
        
        T_tilde = (T - Tm) / (T0-Tm)
        if delayed:
            V_tilde = V_of_T_d(t, T_tilde, A, B, C, M, tdiff_st, N=70)
            m = (1 - V_tilde) * V0 * rhoi
        else:
            V_tilde = V_of_T(t, T_tilde, A, B, C)
            m = (1 - V_tilde) * V0 * rhoi
        
        # plt.plot(t/60, T, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, T_tilde, '.-', color=color, label=f'{mass} kg, {freq} Hz')

        # plt.plot(t/60  , V_tilde, '.-', color=color,  label=f'{mass} kg, {freq} Hz')
        
        # lt = np.where(np.isnan(t))[0][0]
        # plt.plot(t/ t[lt-1] , V_tilde, '.-', color=color,  label=f'{mass} kg, {freq} Hz')

        plt.plot(t/60, m, '.-', color=color,  label=f'{mass} kg, {freq} Hz')

plt.xlabel(r'$t$ (min)')
# plt.ylabel(r'$T$ (°C)')
plt.ylabel(r'$\tilde{V}$ (°C)')

plt.legend()
plt.grid()

plt.show()

#%%

# all data

# tdiff_st = 0.05 # 1/s (diffusivity time of 8.5 mm of steel , alpha/L^2 = (4/1e6) / (0.0085)**2 )
tdiff_st = 1e5 # 1/s (trying values )

delayed = 0

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps/'

names = '*Hz.csv'

file_name = glob.glob(file_path+names)
file_name = file_name[:13]+file_name[14:]

avoid_mass = []

plt.figure()

for name in file_name:
    splits = name[-20:].split('-')[-2:]     
    mass, freq = splits[0][:-2], splits[1][:-6]

    if mass not in avoid_mass:
        start = starts[mass+','+freq]
        color = get_color(mass, freq)
    
        df = pd.read_csv(name, delimiter=";", encoding="ISO-8859-1", header=0)    
        df['Timestamp'] = to_seconds(df["Timestamp"])
    
        t = np.array(df["Timestamp"][:])
        Tt = np.array(df["Water top °C"][:])
        Tb = np.array(df["Water bot °C"][:])
        
        T0 = Tb[0] # °C
        T = T0 + (Tt-Tt[0] + Tb-Tb[0] ) / 2
        t, T = t[start:] - t[start], T[start:] 
        
        Vb = (mass_bath[mass] - 10.)/ rhow # m^3
        V0 = 1.0 * float(mass) / rhoi # m^3
    
        A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, k )    
        # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, 0, k )    
        # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, 0 )    
        
        T_tilde = (T - Tm) / (T0-Tm)
        if delayed:
            V_tilde = V_of_T_d(t, T_tilde, A, B, C, M, tdiff_st, N=70)
            m = (1 - V_tilde) * V0 * rhoi
        else:
            V_tilde = V_of_T(t, T_tilde, A, B, C)
            m = (1 - V_tilde) * V0 * rhoi
        
        # plt.plot(t/60, T, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, T_tilde, '.-', color=color, label=f'{mass} kg, {freq} Hz')

        # plt.plot(t/60  , V_tilde, '.-', color=color,  label=f'{mass} kg, {freq} Hz')
        
        # lt = np.where(np.isnan(t))[0][0]
        # plt.plot(t/ t[lt-1] , V_tilde, '.-', color=color,  label=f'{mass} kg, {freq} Hz')

        plt.plot((t/60)[::50], m[::50], '.', color=color,  label=f'{mass} kg, {freq} Hz', mfc='none', markersize=10)


folder_paths = ['/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Test7-5kg-experiment/Temperature Recordings/',
                '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Test5-10kg-experiment/Temperature Recordings/',
                '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Test6-20kg/Temperature Recordings/' ]
indexes = {0:[1,-2,-4,-2], 1:[1,-2,-3,-2], 2:[-1,-10,-4,-2] }

names = '*.csv'


for j in range(len(folder_paths)):
    file_name = glob.glob(folder_paths[j]+names)
    indx = indexes[j]
    for n in range(len(file_name)):

        splits = file_name[n][:].split('-')
        mass, freq = splits[indx[0]][:indx[1]], splits[indx[2]][:indx[3]]
        
        color = get_color(mass, freq)
    
        df = pd.read_csv(file_name[n], delimiter=";", encoding="ISO-8859-1", header=0)    
        df['Timestamp'] = to_seconds(df["Timestamp"])
    
        t = np.array(df["Timestamp"][:])
        Tt = np.array(df["Water top °C"][:])
        Tb = np.array(df["Water bot °C"][:])
        
        T0 = Tb[0] # °C
        T = T0 + (Tt-Tt[0] + Tb-Tb[0] ) / 2
        t, T = t[start:] - t[start], T[start:] 
        
        Vb = (mass_bath[mass] + 0)/ rhow # m^3
        V0 = 0.97 * float(mass) / rhoi # m^3
    
        A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, k )    
        # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, 0, k )    
        # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, 0 )    
        
        T_tilde = (T - Tm) / (T0-Tm)
        if delayed:
            V_tilde = V_of_T_d(t, T_tilde, A, B, C, M, tdiff_st, N=70)
            m = (1 - V_tilde) * V0 * rhoi
        else:
            V_tilde = V_of_T(t, T_tilde, A, B, C)
            m = (1 - V_tilde) * V0 * rhoi
        
        # plt.plot(t/60, T, '.-', color=color, label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, T_tilde, '.-', color=color, label=f'{mass} kg, {freq} Hz')

        # plt.plot(t/60  , V_tilde, '.-', color=color,  label=f'{mass} kg, {freq} Hz')
        
        # lt = np.where(np.isnan(t))[0][0]
        # plt.plot(t/ t[lt-1] , V_tilde, '.-', color=color,  label=f'{mass} kg, {freq} Hz')

        plt.plot((t/60)[::50], m[::50], '--', color=color,  label=f'{mass} kg, {freq} Hz')


plt.xlabel(r'$t$ (min)')
# plt.ylabel(r'$T$ (°C)')
plt.ylabel(r'$\tilde{V}$ (°C)')

# plt.legend( loc=[1.1,0.0], ncols=2)
# plt.tight_layout()
plt.grid()

savepath = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/'
savename = 'old_v_new(n_corr).pdf'
# plt.savefig(savepath+savename, dpi=200, bbox_inches='tight')

plt.show()


#%%
# Test for calculating Hd
file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Heat transfers-capacity tests/2026-02-09_16.18.28-T-im-system_temp_drop.csv'
df = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1", header=0)

df['Timestamp'] = to_seconds(df["Timestamp"])
 
t_hd = np.array(df["Timestamp"][:])
T_hd = np.array(df["Water bot °C"][:])

plt.figure()
plt.plot(t_hd, T_hd, '.-')
plt.xlabel('t (sec)')
plt.ylabel('T (°C)')
plt.show()


#%%
save = 0

savepath = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/'
savename = 'old_vs_new.pdf'

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps/'
# name = '2026-02-17_09.38.38-T-im-20kg-1Hz.csv'
names = [ '*T-im-10kg-2Hz.csv', '*T-im-10kg-4Hz.csv', '*T-im-10kg-8Hz.csv' ]

plt.figure()
for name in names: 
                
    file_name = glob.glob(file_path+name)
    
    df = pd.read_csv(file_name[0], delimiter=";", encoding="ISO-8859-1", header=0)    
    df['Timestamp'] = to_seconds(df["Timestamp"])
     
    t_hd = np.array(df["Timestamp"][:])
    # T_hd = np.array(df["Water bot °C"][:])
    T_hd = np.array(df["Water top °C"][:])
    T_hd = T_hd-T_hd[0]  
    
    plt.plot(t_hd / 60, T_hd, '.-', label=f'New: {name[-7]} Hz' )
    # plt.plot( T_hd, '.-', label=f'{name[-7]} Hz' )
# plt.xlabel('t (min)')
# plt.ylabel('T (°C)')
# plt.legend()
# plt.show()


file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Test5-10kg-experiment/Temperature Recordings/'
# name = '2026-02-17_09.38.38-T-im-20kg-1Hz.csv'
names = [ '*2Hz*', '*4Hz*', '*8Hz*' ]

# plt.figure()
for name in names: 
                
    file_name = glob.glob(file_path+name)
    
    print('name:', file_name)
    
    df = pd.read_csv(file_name[0], delimiter=";", encoding="ISO-8859-1", header=0)
    
    df['Timestamp'] = to_seconds(df["Timestamp"])

     
    t_hd = np.array(df["Timestamp"][:])
    # T_hd = np.array(df["Water bot °C"][:])
    T_hd = np.array(df["Water top °C"][:])
    T_hd = T_hd-T_hd[0]  
    
    plt.plot(t_hd / 60, T_hd, '--', label=f'Old: {name[-4]} Hz' )
    # plt.plot( T_hd, '.-', label=f'{name[-7]} Hz' )
plt.xlabel('t (min)')
plt.ylabel('T (°C)')
plt.legend()
plt.ylim(11.5-21, 21-21)
plt.grid()
if save: plt.savefig(savepath+savename, dpi=200, bbox_inches='tight')
plt.show()

#%%
# Nu vs Ra
umrss = {'1':0.003957, '2':0.00817, '4':0.01844, '8':0.03881, '12':0.06663}
g = 9.81
Pr = 7
nu, diff_th = 1e-6, 0.143e-6 # m2 / s

def get_adims(V_tilde, V0, T, ctes, Tm=0, plot=False):
    A,B,C,M, alpha,beta,gamma,kappa,Ste = np.array(ctes)
    
    V = V_tilde * V0
    if mass+','+freq == '20,1': Vsv = savgol_filter(V, len(V)//12, 3)
    if mass+','+freq in ['20,2','20,4','20,8','20,12']: Vsv = savgol_filter(V, len(V)//6, 3)
    else: Vsv = savgol_filter(V, len(V)//4, 3)
    
    if plot:
        plt.figure()
        plt.plot(t, V/V0, '.')
        plt.plot(t, Vsv/V0, 'k-')
        plt.show()
        
    mask1 = np.gradient(Vsv)>0
    mask2 = (Vsv/V0)<0.2
    if np.sum(mask1) > 0:
        fin = np.min( [np.where(mask2)[0][0],  np.where(mask1)[0][0]] )
    else:
        fin = np.where(mask2)[0][0]

    # Tsv = savgol_filter(T, 40, 3)
    Rsv = np.cbrt(Vsv)
    gRsv = np.gradient(Rsv,t)
    
    Nu = - beta * Ste * Pr * np.mean(gRsv[:fin]) * Rsv[0] / nu
    Re = u_rms * Rsv[0] / nu
    Ra = g * Rsv[0]**3 / (diff_th * nu) * np.abs( density_millero(0, 0) - density_millero(T[0], 0) ) / density_millero(T[0], 0)
    
    Ste_t = L / (cp * (T-Tm))
    Nut = - beta * Ste_t[:fin] * Pr * gRsv[:fin] * Rsv[:fin] / nu
    Ret = u_rms * Rsv[:fin] / nu
    Rat = g * Rsv[:fin]**3 / (diff_th * nu) * np.abs( density_millero(0, 0) - density_millero(T[:fin], 0) ) / density_millero(T[:fin], 0)
    
    return Nu,Re,Ra,Nut,Ret,Rat

repes = 1
mm100 = 0
mm50 = 1
mm10 = 1

temp_correction = 1
hdfit = 1
ploty = None


avoid_mass = ['15'] #'10','15','20']
avoid_freq = []

colors, sizes = [], []
Nus, Res, Ras = [],[],[]
Nuts, Rets, Rats = [],[],[]

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps/'
names = '*Hz.csv'
file_name = glob.glob(file_path+names)


for name in file_name:
    splits = name[-20:].split('-')[-2:]     
    mass, freq = splits[0][:-2], splits[1][:-6]

    if (mass not in avoid_mass) and (freq not in avoid_freq):
        start = starts[mass+','+freq]
        color = get_color(mass, freq)
        u_rms = umrss[freq]
        
        bathmass = mass_bath[mass] - 10.
        if mass+','+freq != '15,4':
            icemass = mass_ice[mass+','+freq]
        else: icemass = float(mass)
        V0 = icemass / rhoi
        
        t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, temp_correction=temp_correction, delayed=0, Hd_fit=hdfit )

        Nu,Re,Ra,Nut,Ret,Rat = get_adims(V_tilde, V0, T, ctes, Tm=0)
        
        colors.append(color); sizes.append(25)
        Nus.append(Nu); Res.append(Re); Ras.append(Ra)
        Nuts.append(Nut); Rets.append(Ret); Rats.append(Rat)
        
        
file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Repetitions/'
names = '*Hz.csv'
file_name = glob.glob(file_path+names)

avoid_repe = [] # ['1','2']

rep_starts = {'1,5,4':120, '1,5,8':132, '1,5,12':132, '2,5,4':269, '2,5,8':180, '2,5,12':120, '3,5,12':85, '3,5,8':162, '3,5,4':207,
              '4,5,8':382, '2,10,4':260, '4,5,4':213 }

rep_ice = {'1,5,4':4.874, '1,5,8':4.924, '1,5,12':4.907, '2,5,4':5.003, '2,5,8':5.008, '2,5,12':4.980, '3,5,12':4.829, '3,5,8':4.997,
           '3,5,4':4.995, '4,5,8':4.922, '2,10,4':10.125, '4,5,4':4.964 }
        
if repes:
    for name in file_name:
        splits = name[-20:].split('-')[-2:]     
        mass, freq = splits[0][:-2], splits[1][:-6]
    
        if (mass not in avoid_mass) and (freq not in avoid_freq):
            start = starts[mass+','+freq]
            color = get_color(mass, freq)
            u_rms = umrss[freq]
            
            bathmass = mass_bath[mass] - 10.
            if mass+','+freq != '15,4':
                icemass = mass_ice[mass+','+freq]
            else: icemass = float(mass)
            V0 = icemass / rhoi
            
            t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, 
                                                          temp_correction=temp_correction, delayed=0, Hd_fit=hdfit)
    
            Nu,Re,Ra,Nut,Ret,Rat = get_adims(V_tilde, V0, T, ctes, Tm=0)
            
            colors.append(color); sizes.append(25)
            Nus.append(Nu); Res.append(Re); Ras.append(Ra)
            Nuts.append(Nut); Rets.append(Ret); Rats.append(Rat)
            

        
file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps-50mm/'
names = '*Hz.csv'
file_name = glob.glob(file_path+names)

rep_starts = {'1,10,4':444, '2,10,4':330 }
rep_ice = {'1,10,4':9.246, '2,10,4':10.312 }
            
if mm50:
    for name in file_name:
        splits = name[-30:].split('-')[-4:]     
        repe, mass, freq = splits[0][-1], splits[2][:-2], splits[3][:-6]

        if (mass not in avoid_mass) and (freq not in avoid_freq) and (repe not in avoid_repe):
            start = rep_starts[repe+','+mass+','+freq]
            color = get_color(mass, freq)
            # color = get_color(mass, freq)
            u_rms = umrss[freq]
            
            bathmass = mass_bath[mass] - 10.
            if repe+','+mass+','+freq == '1,10,4':
                bathmass = mass_bath[mass] - 5.
            icemass = rep_ice[repe+','+mass+','+freq]
            V0 = icemass / rhoi
            
            t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, 
                                                          temp_correction=temp_correction, delayed=0, Hd_fit=hdfit)
            
            Nu,Re,Ra,Nut,Ret,Rat = get_adims(V_tilde, V0, T, ctes, Tm=0, plot=1)
            
            colors.append(color); sizes.append(50)
            Nus.append(Nu); Res.append(Re); Ras.append(Ra)
            Nuts.append(Nut); Rets.append(Ret); Rats.append(Rat)
            
file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps-10mm/'
names = '*Hz.csv'
file_name = glob.glob(file_path+names)

rep_starts = {'0,10,4':99, '1,10,4':246, '0,10,12':201, '0,10,1':414 }
rep_ice = {'0,10,4':9.471+0.607, '1,10,4':10.065, '0,10,12':9.203, '0,10,1':9.992 }

if mm10:
    for name in file_name:
        splits = name[-30:].split('-')[-4:]     
        repe, mass, freq = splits[0][-1], splits[2][:-2], splits[3][:-6]

        if (mass not in avoid_mass) and (freq not in avoid_freq) and (repe not in avoid_repe):
            start = rep_starts[repe+','+mass+','+freq]
            color = get_color(mass, freq)
            # color = get_color(mass, freq)
            u_rms = umrss[freq]
            
            bathmass = mass_bath[mass] - 10.
            icemass = rep_ice[repe+','+mass+','+freq]
            V0 = icemass / rhoi
            
            t, T, T_tilde, V_tilde, m, ctes = plot_things(name, bathmass, icemass, Hd, k, ploty, start, color, 
                                                          temp_correction=temp_correction, delayed=0, Hd_fit=hdfit)
    
            Nu,Re,Ra,Nut,Ret,Rat = get_adims(V_tilde, V0, T, ctes, Tm=0)
            
            colors.append(color); sizes.append(10)
            Nus.append(Nu); Res.append(Re); Ras.append(Ra)
            Nuts.append(Nut); Rets.append(Ret); Rats.append(Rat)
        
#%%
merke = {25:'o', 10:'^', 50:'s'}
mark = [ merke[m] for m in sizes ]

fig, ax = plt.subplots(1,2, layout='constrained', figsize=(10,4) )

for xi, yi, c,  m in zip(Res, Nus, colors, mark):
    ax[0].scatter(xi, yi, marker=m, c=c)

rree = np.linspace(1.8e3,1.7e4,3)
ax[0].plot( rree, rree * 0.7, 'k--', label=r'$Nu \propto Re$' )

ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel(r'$\langle \text{Re} \rangle$')
ax[0].set_ylabel(r'$\langle \text{Nu} \rangle$')
ax[0].legend()


for i in range(len(Nuts)):
    ax[1].plot( Rets[i], Nuts[i], mark[i], c=colors[i], markersize=3 )

rree = np.linspace(1.8e3,1.7e4,3)
ax[1].plot( rree, rree * 0.7, 'k--', label=r'$Nu \propto Re$' )

ax[1].legend()
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel(r'Re$(t)$')
ax[1].set_ylabel(r'Nu$(t)$')

savepath = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/MMMM-april26/'
savename = 'nu_re_sizes(extra).png'
# plt.savefig(savepath+savename, dpi=100, bbox_inches='tight')

plt.show()

plt.figure()
for i in range(len(Nuts)):
    plt.plot( Rats[i], Nuts[i], '.-', c=colors[i] )

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Ra')
plt.ylabel('Nu')
plt.show()

#%%
# =============================================================================
# Data with errors
# =============================================================================

import numpy as np 
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
from scipy.integrate import cumulative_trapezoid
import time

# from matplotlib import cm
# from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, least_squares
# from scipy.stats import linregress

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

def bt_nm(lam,t,T, tdiff):
    s = 2/lam
    beta = tdiff * lam**2
    ct = exp_convolution(t, T, beta)
    t1 = np.exp(-beta*t) 
    return s/(lam) * ( t1 - T + beta * ct )

def energy_delayed(t,T, tdiff, N=1000):

    lam = np.pi/2 * (1 + 2* np.arange(0,N)) 
    mb = np.zeros_like(t)
    for n in range(0,N):        
        mbn = bt_nm(lam[n], t, T, tdiff) 
        
        mb += mbn
        
    return mb


def to_seconds(timestamps, fmt="%Y-%m-%d_%H.%M.%S.%f"):
    t0 = datetime.strptime(str(timestamps[0]), fmt)
    
    times = []
    for t in timestamps:
        strt = str(t)
        if len(strt) > 3: times.append( (datetime.strptime(str(t), fmt)-t0).total_seconds() )
        elif len(strt) == 3: times.append(np.nan)

    return np.array(times)

def constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L, Hd, k ):
    alpha = Hd / (rhow*Vo*cp)
    beta = rhoi / rhow
    gamma = Vb/Vo
    kappa = k / (rhow*Vo*cp)
    Ste = L / (cp * (To-Tm))

    A = beta/(gamma+alpha)
    B = A * Ste
    C = kappa /(gamma+alpha)
    M = alpha / (gamma+alpha)

    return A,B,C,M, alpha,beta,gamma,kappa,Ste    


def rk4_sol(dt, A,B,C,D):
    
    def func(t,R):
        top = B * (R**3 -1) + 1 + C/D * (R+t-1)
        bot = A * (R**3 -1) - 1
        return top/bot
    
    R_sol = [1]
    t = [0]

    # for n in range(1,len(t)):
    while R_sol[-1] > 0:
        
        rn, tn = R_sol[n-1], t[-1]
        k1 = func(tn, rn)
        k2 = func(tn + dt/2, rn + k1 * dt/2)
        k3 = func(tn + dt/2, rn + k2 * dt/2)
        k4 = func(tn + dt, rn + k3 * dt)
        
        rnext = rn + dt/6 * (k1+2*k2+2*k3+k4)
        
        R_sol.append( rnext )
        t.append( tn + dt )    
    
    return np.array(R_sol), np.array(t)

def solution_R( R, A,B ):
    pterm = A*R/B
    prod = (A+B) / (6 * np.cbrt(B-1)**2 * np.cbrt(B)**4)
    arcterm = 2*np.sqrt(3) * np.arctan( (1+2*R * np.cbrt(B/(B-1)) ) / np.sqrt(3) )
    logterm = np.log( ( (np.cbrt(B-1) + np.cbrt(B)*R)**2 - np.cbrt(B-1) * np.cbrt(B) * R ) / (np.cbrt(B-1) - np.cbrt(B)*R )**2  )
    return pterm + prod * (arcterm + logterm)


def V_of_T(t,T, A,B,C):
    t1 = 1 - (1-T)/(B+A*T)
    inte = cumulative_trapezoid(1-T, t, initial=0)
    return t1 - C/(B+A*T) * inte

def V_of_T_d(t,T, A,B,C, M, tdiff, N=1000):
    t1 = 1 - (1-T)/(B+A*T)
    t2 = M * energy_delayed(t, T, tdiff, N=N) /(B+A*T)
    inte = cumulative_trapezoid(1-T, t, initial=0)
    return t1 + t2 - C/(B+A*T) * inte

def energies(t,T, alpha,beta,gamma,kappa,Ste):
    eb = 1-T
    ed = alpha/gamma * (1-T)
    el = kappa/gamma * cumulative_trapezoid(1-T, t, initial=0)

    A = beta/(gamma+alpha)
    B = A * Ste
    C = kappa /(gamma+alpha)
    V = V_of_T(t, T, A, B, C)
    
    elat = beta/gamma * Ste * (1-V)
    emelt = beta/gamma (1-V) * T

    return eb,ed,el, elat, emelt
    
def get_color(mass, freq):
    folder_colors = {'5': cm.Blues, '10': cm.Greens, '20': cm.Reds, '15': cm.Greys }
    frequencies = ['1','2','4','8','12']

    cmap = folder_colors[mass]
    freq_index = frequencies.index(freq)
    return cmap(0.3 + 0.7 * freq_index / (len(frequencies) - 1))  # soft to dark

def average_window( x, window):
    """
    Parameters
    ----------
    t : 1D-array
        time.
    T : 1D-array
        temperature.
    window : int
        window size to average in.

    Returns
    -------
    t : 1D-array
        averaged time.
    T : 1D-array
        averaged temperature.
    """
    xa = np.convolve(x, np.ones(window)/window, mode='valid')
    return xa


def constants_err( T0, Tm, m0, mb, rhoi, rhow, cp, L, Hd, k, errs, N ):
    """
    Calculates the parameters A, B and C with errors
    
    errs : list
        error of each of the previous variables (in the same other as the function takes them).
    N : int
        number of samples to create.
    """
    err_T0, err_Tm, err_m0, err_mb, err_rhoi, err_rhow, err_cp, err_L, err_Hd, err_k = errs
    
    T0_mc = np.random.normal(T0, err_T0, N)
    Tm_mc = np.random.normal(Tm, err_Tm, N)
    V0_mc = 1.0 * np.random.normal(m0, err_m0, N) / rhoi
    Vb_mc = 1.0 * np.random.normal(mb, err_mb, N) / rhow 
    rhoi_mc = np.random.normal(rhoi, err_rhoi, N)
    rhow_mc = np.random.normal(rhow, err_rhow, N)
    cp_mc = np.random.normal(cp, err_cp, N)
    L_mc = np.random.normal(L, err_L, N)
    Hd_mc = np.random.normal(Hd, err_Hd, N)
    k_mc = np.random.normal(k, err_k, N)

    
    A_mc, B_mc , C_mc, M, alpha,beta,gamma,kappa,Ste = constants( T0_mc, Tm_mc, V0_mc, Vb_mc, rhoi_mc, rhow_mc, cp_mc, L_mc, Hd_mc, k_mc )    
    
    return A_mc, B_mc, C_mc, T0_mc, Tm_mc


def V_of_T_error( T, err_T, T0_mc, Tm_mc , A,B,C, N, chunk =1000, show_bar=False):
    """
    Parameters
    ----------
    T : 1d array
        list with the array of T (dimensional).
    A : 1d-array
        variable A with N samples.
    B : 1d-array
        variable B with N samples.
    C : 1d-array
        variable C with N samples.
    N : int
        number of samples.
    chunk : int
        Size of chunks to do calculations, should be a divsor of N. Default = 1000

    Returns
    -------
    V_mean : (len(T)) array
        Volume mean values .
    V_std : (len(T)) array
        Volume standard deviation.

    """
    T0, Tm = T0_mc, Tm_mc
    V_err = np.empty((len(T), N))

    for i in tqdm(range(0, N, chunk), disable=not show_bar):
        j = i + chunk
        samples = np.random.normal(T[:, None], err_T, size=(len(T), j - i))
        samples[0,:] = T0[i:j]
        
        Tn = (samples - Tm[None, i:j]) / (T0[None, i:j] - Tm[None, i:j])
        
        den = B[None, i:j] + A[None, i:j] * Tn
        t_1 = 1 - (1 - Tn) / den
        inte = cumulative_trapezoid(1 - Tn, t, axis=0, initial=0)
        
        V_err[:, i:j] = t_1 - (C[None, i:j] / den) * inte

    return np.mean(V_err,axis=1), np.std(V_err,axis=1)
    

#------ Thermocuple calibration ------ 
def temperature_calibration( fit_deg ):
    file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Cal thermo/calibration_pt104.csv'
    df = pd.read_csv(file_path, delimiter=",", encoding="ISO-8859-1", header=0)
    t_pt100, T_pt100 = to_seconds(df['Unnamed: 0'], fmt="%H:%M:%S"), np.array(df['Channel 1 Ave. (C)'])
    
    file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Cal thermo/2026-03-13_14.59.48.csv'
    df = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1", header=0)
    df['Timestamp'] = to_seconds(df["Timestamp"], fmt="%Y-%m-%d_%H.%M.%S.%f")
    t_th = np.array(df["Timestamp"][:]) + 48
    Tb_th, Tt_th = np.array(df["Water bot °C"][:]), np.array(df["Water top °C"][:])
    Tmean_th = ( Tb_th + Tt_th ) / 2
    
    T_pi_int = np.interp(t_th, t_pt100, T_pt100)
    end_stair = 308 * 60
    fil_th = t_th < end_stair 

    fit_mean = np.polyfit( Tmean_th[fil_th], T_pi_int[fil_th], fit_deg)
    return fit_mean
    
def correct_temperature( T_mean, coeffs ):
    return np.polyval(coeffs, T_mean )


#------ Definitions and values ------ 
starts = {'20,4':198, '20,2':139, '10,12':140, '10,8':95, '10,4':125, '10,2':240, '10,1':300, '5,12':165, '5,8':90, '5,4':130,
          '5,2':170, '5,1':120, '20,1':160, '20,12':30, '20,8':35, '15,4':0 }

mass_ice = {'5,1':4.992, '5,2':4.997, '5,4':5.009, '5,8':5.011, '5,12':5.024, '10,1':4.977+5.015, '10,2':5.037+5.006, '10,4':5.011+5.036, 
            '10,8':5.013+5.024, '10,12':4.997+5.008, '20,1':5.015+4.970+4.965+4.949, '20,2':5.015+4.970+4.965+4.949, '20,4':5.016+5.015+5.001+4.987, 
            '20,8':5.052+5.021+5.013+4.931, '20,12':5.011+5.023+5.010+5.010}

mass_bath = {'10':112, '5':117, '20':102, '15':107}

rhow, rhoi = 998.2, 916.8 # kg/m3
Tm = 0 #°C
L = 334000 # J/kg 
cp = 4184 # J/(kg K)

k = 11.55 # J/sK
Hd = 34320 #J/K
Tm = 0 #°C

calibration_deg = 2
coeffs_cal = temperature_calibration( calibration_deg )
correct_temp = lambda T_mean: correct_temperature( T_mean, coeffs_cal )

# Errors on variables 
err_T, err_Tm = 0.1, 0   # error for T0 is equal to the error in T
err_m0, err_mb = 0.01, 0.5
err_rhoi, err_rhow = 0, 0
err_cp, err_L = 0, 0
err_Hd, err_k = 4000, 0.1

all_errors = [err_T, err_Tm, err_m0, err_mb, err_rhoi, err_rhow, err_cp, err_L, err_Hd, err_k]

#%%

# tdiff_st = 0.05 # 1/s (diffusivity time of 8.5 mm of steel , alpha/L^2 = (4/1e6) / (0.0085)**2 )
# tdiff_st = 1e-4 # 1/s (trying values )

statistics = 10000

delayed = 0 # don't use in here (keep as 0)
temp_correction = 1

averaged = 0
window = 10 # 10 is one second

montecarlo = 0
Hd_fit = 1

# avoid_mass = ['15','5','10']
# avoid_freq = ['1','2','8','12']

avoid_mass = ['15']
avoid_freq = []

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Temps/'
names = '*Hz.csv'
file_name = glob.glob(file_path+names)

maplot = {'5':0, '10':1, '20':2, '15':None}
fig, ax = plt.subplots(1,3, figsize = (15,4), sharey=True, layout='constrained')

for name in file_name:
    splits = name[-20:].split('-')[-2:]     
    mass, freq = splits[0][:-2], splits[1][:-6]

    if (mass not in avoid_mass) and (freq not in avoid_freq):
        start = starts[mass+','+freq]
        color = get_color(mass, freq)
        splot = maplot[mass]
        
        if mass+','+freq != '15,4':
            icemass = mass_ice[mass+','+freq]
        else: icemass = float(mass)
    
        df = pd.read_csv(name, delimiter=";", encoding="ISO-8859-1", header=0)    
        df['Timestamp'] = to_seconds(df["Timestamp"])
    
        t = np.array(df["Timestamp"][:])
        Tt = np.array(df["Water top °C"][:])
        Tb = np.array(df["Water bot °C"][:])
        
        if temp_correction:
            Tmean = correct_temp( (Tt + Tb) / 2 )
            t, T = t[start:] - t[start], Tmean[start:] 
            T0 = Tmean[start]
            # plt.title('With tempretature correction')
        else:
            T0 = Tb[0] # °C # maybe should use Tt
            T = T0 + (Tt-Tt[0] + Tb-Tb[0] ) / 2
            t, T = t[start:] - t[start], T[start:] 
            # plt.title('Without tempretature correction')

        # for j in range(statistics):
            
        Vb = (mass_bath[mass] - 10.)/ rhow # m^3
        V0 = 1.0 * icemass / rhoi # m^3
        
        T_tilde = (T - Tm) / (T0-Tm)
    
    
        if Hd_fit:
            def fit_hd(val): #Using minimum ±0.2 min
                A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0 * 1.0, Vb, rhoi, rhow, cp, L, val[0], k )    
                V_tilde = V_of_T(t, T_tilde, A, B, C)
                minar = np.nanargmin(V_tilde)
                minmean = V_tilde[minar-120:minar+120]  #np.nanmean( V_tilde[minar-120:minar+120] )
                return  minmean[~np.isnan(minmean)] 
            ls = least_squares(fit_hd, [Hd], bounds=((0.),(Hd*1.5)), method='trf')
            # ls = least_squares(fit_hd, [Hd], method='lm')
    
            A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0 * 1.0, Vb, rhoi, rhow, cp, L, ls.x[0], k )
            print( ls.x )
    
        else:
            A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0 * 1.0, Vb, rhoi, rhow, cp, L, Hd, k )
            
        # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, 12.7 )    
        # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, 500*30, 11.7 )    
        # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, 0, 0 )    
        
        
        #---- Montecarlo ----
        if montecarlo:
            all_errors[2] = err_m0 * float(mass)/5 # correction to error in initial ice mass (since the value is valid for 5 kg)
            A_mc, B_mc, C_mc, T0_mc, Tm_mc = constants_err( T0, Tm, icemass, mass_bath[mass]-10., rhoi, rhow, cp, L, Hd, k, all_errors, statistics )    
            V_me, V_sd = V_of_T_error( T, err_T, T0_mc, Tm_mc, A_mc,B_mc,C_mc, statistics, chunk= np.min([statistics//10,1000]), show_bar=False)        

        if delayed:
            V_tilde = V_of_T_d(t, T_tilde, A, B, C, M, tdiff_st, N=70)
            m = (1 - V_tilde) * V0 * rhoi
        else:
            V_tilde = V_of_T(t, T_tilde, A, B, C)
            m = (1 - V_tilde) * V0 * rhoi
            
        if averaged:
            t, m = average_window(t, window), average_window(m, window)
            T_tilde, V_tilde = average_window(T_tilde, window), average_window(V_tilde, window)
                    
        
        ax[splot].plot(t/60, V_tilde, '.-', color=color,  label=f'{mass} kg, {freq} Hz')
        # plt.plot(t/60, V_me, '.', color=color,  label=f'{mass} kg, {freq} Hz')

        if montecarlo:
            ax[splot].fill_between(t/60, V_me-V_sd, V_me+V_sd,  color=color, alpha=0.5)#, zorder=0.6)

        # plt.plot(t/60, m , '.-', color=color,  label=f'{mass} kg, {freq} Hz')

for i in range(3):
    ax[i].set_xlabel(r'$t$ (min)')
    # ax[i].set_ylabel(r'$T$ (°C)')
    
    ax[i].legend()
    ax[i].grid()

ax[0].set_ylabel(r'$\tilde{V}$')
ax[0].set_ylim(-0.13,1.05)

savepath = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/MMMM-april26/'
savename = 'exps_hdfit.png'
plt.savefig(savepath+savename, dpi=100, bbox_inches='tight')


plt.show()

#%%
# Tests
Hd = 0
k = 0
statistics = 1000

err_T, err_Tm = 0.2, 0   # error for T0 is equal to the error in T
err_m0, err_mb = 0.03, 0.03
err_rhoi, err_rhow = 0, 0
err_cp, err_L = 0, 0
err_Hd, err_k = 0, 0.0

all_errors = [err_T, err_Tm, err_m0, err_mb, err_rhoi, err_rhow, err_cp, err_L, err_Hd, err_k]


delayed = 0 # don't use in here (keep as 0)
temp_correction = 1

averaged = 0
window = 10 # 10 is one second

avoid_mass = ['5','15','10']
# avoid_freq = ['1','2','8','12']
# avoid_mass = []
avoid_freq = []

file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Test/'
names = '*.csv'
file_name = glob.glob(file_path+names)

mass_baths = [ 18.153, 14.999, 14.999, 15.003]
icemasss = [1.566, 1.191, 1.379, 1.280]

plt.figure()

for j,name in enumerate(file_name[:]):
    if j > 0 and j<4:
        start = 63
        
        df = pd.read_csv(name, delimiter=";", encoding="ISO-8859-1", header=0)    
        df['Timestamp'] = to_seconds(df["Timestamp"])
    
        t = np.array(df["Timestamp"][:])
        Tt = np.array(df["Water top °C"][:])
        Tb = np.array(df["Water bot °C"][:])
        
        if temp_correction:
            Tmean = correct_temp( (Tt + Tb) / 2 )
            t, T = t[start:] - t[start], Tmean[start:] 
            T += 0.
            T0 = T[start]
            plt.title('With tempretature correction')
        else:
            T0 = Tb[0] # °C # maybe should use Tt
            T = T0 + (Tt-Tt[0] + Tb-Tb[0] ) / 2
            t, T = t[start:] - t[start], T[start:] 
            plt.title('Without tempretature correction')
    
        mass_bath = mass_baths[j]
        icemass = icemasss[j]
        
        Vb = mass_bath / rhow # m^3
        V0 = icemass / rhoi # m^3
        
        T_tilde = (T - Tm) / (T0-Tm)
    
        A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, k )    
        # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, 0, 0 )
        # A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, 500 * 28, 0 )
        
        
        #---- Montecarlo ----
        all_errors[2] = err_m0  # correction to error in initial ice mass (since the value is valid for 5 kg)
        A_mc, B_mc, C_mc, T0_mc, Tm_mc = constants_err( T0, Tm, icemass, mass_bath, rhoi, rhow, cp, L, Hd, k, all_errors, statistics )    
        V_me, V_sd = V_of_T_error( T, err_T, T0_mc, Tm_mc, A_mc,B_mc,C_mc, statistics, chunk = np.min([statistics//10,1000]), show_bar=False)        
    
        if delayed:
            V_tilde = V_of_T_d(t, T_tilde, A, B, C, M, tdiff_st, N=70)
            m = (1 - V_tilde) * V0 * rhoi
        else:
            V_tilde = V_of_T(t, T_tilde, A, B, C)
            m = (1 - V_tilde) * V0 * rhoi
            
        if averaged:
            t, m = average_window(t, window), average_window(m, window)
            T_tilde, V_tilde = average_window(T_tilde, window), average_window(V_tilde, window)
                    
        ttf = (mass_bath + icemass * L / (cp*(0-T0)) ) / (mass_bath+icemass)
        print('Final temp (dim) =', ( cp * mass_bath * T0 - icemass * L ) / ( cp * (mass_bath+icemass)) )
        print('Final temp (adim) =', ttf )
        
        # plt.plot(t/60, T, '.-', label='Temperature (dim)')
        # plt.plot(t/60, T_tilde , '.-', label='Temperature (adim)')
        # plt.plot(t/60, T_tilde / ttf , '.-', label='Temperature (adim) / Final temp')
        plt.plot(t/60, V_tilde, '.-', label='Volume (adim)')
        # plt.plot(t/60, m, '.-', label='mass (kg)')
        
        # plt.plot(t/60, V_tilde, '.-',  label='Vol (adim)')
        # plt.plot(t/60, V_me, '.',   label=f'{mass} kg, {freq} Hz')
    
        plt.fill_between(t/60, V_me-V_sd, V_me+V_sd, alpha=0.2, zorder=0.6)
    
            # plt.plot(t/60, m , '.-', color=color,  label=f'{mass} kg, {freq} Hz')

plt.xlabel(r'$t$ (min)')
# plt.ylabel(r'$T$ (°C)')
# plt.ylabel(r'$\tilde{V}$')
# plt.ylim(-.1,1.1)

plt.legend()
plt.grid()

plt.show()

#%%

Hd = 0
k = 0
statistics = 1000
start=1

err_T, err_Tm = 0.1, 0   # error for T0 is equal to the error in T
err_m0, err_mb = 0.02, 0.02
err_rhoi, err_rhow = 0, 0
err_cp, err_L = 0, 0
err_Hd, err_k = 0, 0.0

all_errors = [err_T, err_Tm, err_m0, err_mb, err_rhoi, err_rhow, err_cp, err_L, err_Hd, err_k]


mass_bath = 15.003
icemass = 1.280

plt.figure()


file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Test/'
name = file_path + '2026-03-25_15.47.12-Plastic-tank-withpt100.csv'

df = pd.read_csv(name, delimiter=";", encoding="ISO-8859-1", header=0)    
df['Timestamp'] = to_seconds(df["Timestamp"])

t = np.array(df["Timestamp"][:])
Tt = np.array(df["Water top °C"][:])
Tb = np.array(df["Water bot °C"][:])

Tmean = correct_temp( (Tt + Tb) / 2 )
t, T = t[start:] - t[start], Tmean[start:] 
T += 0.
T0 = T[start]


Vb = mass_bath / rhow # m^3
V0 = icemass / rhoi # m^3

T_tilde = (T - Tm) / (T0-Tm)

A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, k )    

#---- Montecarlo ----
all_errors[2] = err_m0  # correction to error in initial ice mass (since the value is valid for 5 kg)
A_mc, B_mc, C_mc, T0_mc, Tm_mc = constants_err( T0, Tm, icemass, mass_bath, rhoi, rhow, cp, L, Hd, k, all_errors, statistics )    
V_me, V_sd = V_of_T_error( T, err_T, T0_mc, Tm_mc, A_mc,B_mc,C_mc, statistics, chunk = np.min([statistics//10,1000]), show_bar=False)        

V_tilde = V_of_T(t, T_tilde, A, B, C)
m = (1 - V_tilde) * V0 * rhoi

ttf = (mass_bath + icemass * L / (cp*(0-T0)) ) / (mass_bath+icemass)
print('Final temp (dim) =', ( cp * mass_bath * T0 - icemass * L ) / ( cp * (mass_bath+icemass)) )
print('Final temp (adim) =', ttf )

# plt.plot(t/60, T, '.-', label='Temperature (thermo)')
# plt.plot(t/60, T_tilde , '.-', label='Temperature (adim)')
# plt.plot(t/60, T_tilde / ttf , '.-', label='Temperature (adim) / Final temp')
plt.plot(t/60, V_tilde, '.-', label='Volume (thermo)')
# plt.plot(t/60, m, '.-', label='mass (kg)')


file_path = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Go pro-new experiments/Everything/Test/'
name = file_path + 'Pt100-test.csv'

df = pd.read_csv(name, delimiter=",", encoding="ISO-8859-1", header=0)    
df['Unnamed: 0'] = to_seconds( df['Unnamed: 0'], fmt="%H:%M:%S.%f" )


t = np.array(df['Unnamed: 0'][:])
Tpt = np.array(df['Channel 1 Ave. (C)'][:])

t, T = t[start:] - t[start], Tpt[start:] 
t += 60 * 0.15
T0 = Tpt[start]


Vb = mass_bath / rhow # m^3
V0 = icemass / rhoi # m^3

T_tilde = (T - Tm) / (T0-Tm)

A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, k )    

#---- Montecarlo ----
all_errors[2] = err_m0  # correction to error in initial ice mass (since the value is valid for 5 kg)
A_mc, B_mc, C_mc, T0_mc, Tm_mc = constants_err( T0, Tm, icemass, mass_bath, rhoi, rhow, cp, L, Hd, k, all_errors, statistics )    
V_me, V_sd = V_of_T_error( T, err_T, T0_mc, Tm_mc, A_mc,B_mc,C_mc, statistics, chunk = np.min([statistics//10,1000]), show_bar=False)        

V_tilde = V_of_T(t, T_tilde, A, B, C)
m = (1 - V_tilde) * V0 * rhoi

ttf = (mass_bath + icemass * L / (cp*(0-T0)) ) / (mass_bath+icemass)
print('Final temp (dim) =', ( cp * mass_bath * T0 - icemass * L ) / ( cp * (mass_bath+icemass)) )
print('Final temp (adim) =', ttf )

# plt.plot(t/60, T, '.-', label='Temperature (pt)')
# plt.plot(t/60, T_tilde , '.-', label='Temperature (adim)')
# plt.plot(t/60, T_tilde / ttf , '.-', label='Temperature (adim) / Final temp')
plt.plot(t/60, V_tilde, '.-', label='Volume (pt)')
# plt.plot(t/60, m, '.-', label='mass (kg)')


plt.xlabel(r'$t$ (min)')
# plt.ylabel(r'$T$ (°C)')
# plt.ylabel(r'$\tilde{V}$')
# plt.ylim(-.1,1.1)

plt.legend()
plt.grid()

plt.show()

#%%
# Vb = (mass_bath[mass] - 10.)/ rhow # m^3
# V0 = 1.0 * float(mass) / rhoi # m^3

tr = np.random.normal(T0, 0.1, 50000)
vr = 1.0 * np.random.normal(mass_bath[mass]-10., 0.5, 50000) / rhoi 

# A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( T0, Tm, V0, Vb, rhoi, rhow, cp, L, Hd, k )    
A,B,C,M, alpha,beta,gamma,kappa,Ste = constants( tr, Tm, V0, vr, rhoi, rhow, cp, L, Hd, k )    




plt.figure()

# plt.hist(tr, bins=50)
# plt.hist(vr * rhoi, bins=50)
# plt.hist(C, bins=50)

# plt.plot(tr, '.', markersize=1)
# plt.plot(vr * rhoi, '.', markersize=1)

plt.plot(t, T_tilde, '.-')

plt.show()

print(T0)
print(np.mean(tr), np.std(tr))

A, B, C
#%%

#%%
sigma = 0.1
n = 10000

t1 = time.time()
chunk = 1000
V_err = np.empty((len(T), n))

for i in tqdm(range(0, n, chunk)):
    j = i + chunk
    samples = np.random.normal(T[:, None], sigma, size=(len(T), j - i))
    Tn = samples / T0
    
    den = B[None, i:j] + A[None, i:j] * Tn
    t_1 = 1 - (1 - Tn) / den
    inte = cumulative_trapezoid(1 - Tn, t, axis=0, initial=0)
    
    V_err[:, i:j] = t_1 - (C[None, i:j] / den) * inte
t4 = time.time()

print(V_err.shape)
print(t4-t1)

#%%

plt.figure()
# plt.plot(t, T_tilde, '-')
# plt.plot(t, samples[:,0] / T0, '.', markersize=1)
# plt.plot(t, samples[:,10] / T0, '.', markersize=1)

plt.plot(t, V_of_T(t, T_tilde, np.mean(A), np.mean(B), np.mean(C)), '-', zorder=10 ) 
# plt.plot(t, V_err[:,0] )
# plt.plot(t, V_err[:,1] )
# plt.plot(t, V_err[:,2] )
plt.errorbar(t, np.mean(V_err, axis=1), yerr=np.std(V_err, axis=1))

# plt.hist(V_err[0,:], bins=30)
# plt.hist(V_err[1000,:], bins=30)

plt.grid()
plt.show()

#%%
N = 100000

m, hh, c1,c2 = 122, 34320, 2.128e-5, 2.112e-5

k1,k2 = c1*(m*cp+hh), c2*(m*cp+hh)

print(k1, k2)

mc, hhc = np.random.normal(m,0.5, N), np.random.normal(hh,4000, N) 
c1c,c2c = np.random.normal(c1, 6.86e-10, N), np.random.normal(c2, 9.42e-10, N) 

k1c,k2c = c1c*(mc*cp+hhc), c2c*(mc*cp+hhc)

print( np.mean(k1c), np.std(k1c) ) 
print( np.mean(k2c), np.std(k2c) ) 
print( np.mean(k2c/2+k1c/2), np.std(k2c/2+k1c/2) ) 


# plt.figure()
# plt.hist(hfdc, bins=50)
# plt.show()


#%%

cpw = 4184 # J/(kg K)
mw = 7.562
tip = 68
tf = 66.5
tid = 20


hss = cpw * mw *(tip - tf) / (tf - tid)

hss, hss/1500



