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

def R_of_T(T, A,B, set_tot_cero=True):
    bot = A * (T-1) + (B+A)
    argum = 1 - (1-T)/bot 
    if set_tot_cero:
        argum[argum<0] = 0
    return  np.cbrt(argum)

def V_of_T(T, A,B, set_tot_cero=True):
    bot = A * (T-1) + (B+A)
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
masses = {5:117, 10:112, 20:102}

plt.figure(figsize=(14, 8))

for i, label in enumerate(sorted(results.keys(), key=sort_key)):
    folder_name, freq = label.split()
    data = results[label]

    t = data["t_avg"]
    T = data["T_top_avg"]
    T_init = data["T_top_init"]

    mass = float(re.split('-| ',label)[1][:-2])
    Vo = mass / rhoi # m3
    To = T_init #°C
    Vb = masses[mass]/rhow #0.102 # m3        

    A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
    
    minv= np.min(V_of_T((T-Tm)/(To-Tm),A,B, set_tot_cero=False))
    if minv < 0:
        Vo = (1 - minv) * Vo
    A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )



    # Plot main temperature line
    # plt.plot(t, T,
    plt.plot(t, V_of_T((T-Tm)/(To-Tm),A,B),
             label=rf"{label} ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
             color=get_color(folder_name, freq),
             linestyle='-',
             marker='o', markersize=8,
             markeredgecolor='black', linewidth=1)

    # --- Identify and plot the minimum point ---
    min_idx = np.argmin(T)
    t_min = t[min_idx]
    T_min = T[min_idx]
    
    Tsv = savgol_filter(T, len(T)//6, 3)

    # Plot star marker at minimum
    # plt.plot(t, Tsv, 'y--')
    plt.plot(t, V_of_T((Tsv-Tm)/(To-Tm),A,B), 'y--')
    
    # plt.plot(t_min, T_min, marker='*', color='yellow',
    #          markersize=14, markeredgecolor='black', linewidth=1, zorder=5, label=None)

    # --- Print the minimum to terminal ---
    # print(f"⭐ {label}: Min Temperature = {T_min:.2f}°C at {t_min:.1f} s")

fsize = 15
plt.xlabel("Time (s)", fontsize=fsize)
plt.ylabel("Top Temperature (°C)", fontsize=fsize)
plt.title("Top Water Temperature vs Time (Minima Marked)", fontsize=fsize)
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Folder + Frequency", fontsize=fsize)
plt.tight_layout()

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/temperatures.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()

#%%
rhow, rhoi = 998.2, 916.8 # kg/m3
Tm = 0 #°C
L = 334000 # J/kg 
cp = 4184 # J/(kg K)

R = np.linspace( 0,1, 1000 )

masses = {5:117, 10:112, 20:102}
cutoffs = {5:0.5, 10:0.61, 20:1.1}
urmss = {1:0.003957, 2:0.00817, 4:0.01844, 8:0.03881, 12:0.06663}

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
        A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
        Pr = 7
        
        minv= np.min(V_of_T((T-Tm)/(To-Tm),A,B, set_tot_cero=False))
        # if minv < -0:
        #     Vo = (1 - minv) * Vo
        # print( '\t Real mass: {:.4f}kg,'.format(Vo * rhoi), end=' ' )
        
        A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
        
        ctheory = 0.001 * Ste * Pr / beta * urmss[frequency] / np.cbrt(Vo)
        
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
    
    
        plt.plot(t * c, T_linear, label=f'{frequ} Hz',
                  # label=rf"{mass}kg, {freq} ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                  color=get_color(folder_name, freq),
                  linestyle='-',
                  # marker='o', markersize=8, markeredgecolor='black', linewidth=1)
                  marker='o', markersize=8) 
        
        # plt.plot(t, Tsv_linear, 'y--')
        
        plt.plot( t * c , lin_cons(t, ls.x) , 'm--' )
        
        # plt.plot(t,V_tilde,'.-')

 
# plt.hlines(cutoffs[mass_fig[0]], 0, 500, color='black', linestyles='dashed', label='Fit limit' ) #plot limit of fit 
plt.plot([0,2],[0,2],'--',color='y')

fsize = 12
plt.xlabel(r"$t$ (s)", fontsize=fsize )
plt.ylabel(r"$R_s(R_T(T)) - R_s(1)$", fontsize=fsize )
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
        
        A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )

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

ax[1].plot( [1,2,4,8,12], rhow/rhoi*cp/L * umrss / Pr * 0.7 , '^', color='orange', label='Theory',zorder=4)

ax[0].legend()
ax[1].legend()

ax[0].set_ylabel(r'$C$ (1/s)')
ax[0].set_xlabel(r'$f$ (Hz)')

ax[1].set_ylabel(r'$C \,\, V_0^{1/3} \, / \, (T_0-T_m)$ (m/sK)')
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

R = np.linspace( 0,1, 1000 )

masses = {5:117, 10:112, 20:102}
cutoffs = {5:0.5, 10:0.61, 20:1.1}
urmss = {1:0.003957, 2:0.00817, 4:0.01844, 8:0.03881, 12:0.06663}

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
    
            A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
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
            A,B, beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
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
plt.savefig(filename, dpi=200, bbox_inches='tight')
plt.show()
# plt.close('all')




#%%
# =============================================================================
# Towards Nu vs Re
# =============================================================================
# Corrected model?

def constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L ):
    beta = rhoi / rhow
    gamma = Vb/Vo    
    Ste = L / (cp * (To-Tm))
        
    return beta, gamma, Ste


def V_of_T(T,To,Tm, beta,gamma,Ste ):
    bot = beta * ( Ste*(To-Tm) + 2*T-Tm-To ) 
    top = gamma * (T-To)
    return 1 + top / bot

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

masses_bath = {5:117, 10:112, 20:102} #in kg

apply_heat_loss = False
show_minima = True

sav_gol_fil = True
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

    beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )      
    V = V_of_T(T, To,Tm, beta,gamma,Ste)
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

    # Plot radius and volume over time (with this energy balance)
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

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

plt.tight_layout()
# h, l = ax[1].get_legend_handles_labels()
# plt.legend(h,l,loc='upper center', ncols=5, bbox_to_anchor=(0.5, -0.5), fancybox=False, shadow=False, fontsize=fsize)
ax[1].legend(loc='upper center', ncols=5, bbox_to_anchor=(0.5, 1.2), fancybox=False, shadow=False, fontsize=fsize)
fig.subplots_adjust(bottom=0.08, top=0.85)

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/experiemtns.pdf'
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
    
    beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )      
    V = V_of_T(T, To,Tm, beta,gamma,Ste)
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

compensate = 0

umrss = {1:0.003957, 2:0.00817, 4:0.01844, 8:0.03881, 12:0.06663} #m/s, u_rms
masses_bath = {5:117, 10:112, 20:102} #in kg

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
    
    beta,gamma,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )      
    V = V_of_T(T, To,Tm, beta,gamma,Ste)
    if apply_heat_loss:
        V -= V_eloss_term(t,T,To,Tm, beta,gamma,Ste, rhow,Vo,cp,m,b)
        
    R = np.cbrt(V)

    Vsv = savgol_filter(V, len(V)//4, 3)
    mask1 = np.gradient(Vsv)>0
    mask2 = Vsv<0.1
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
    ax[0].set_ylabel(rf"$\langle$Nu$\rangle$ / $\langle$Re$\rangle$", fontsize=fsize)
    ax[1].set_ylabel(rf'Nu(t) / Re(t) ', fontsize=fsize)
else:
    ax[0].set_ylabel(rf"$\langle$Nu$\rangle$ / $\langle$Re$\rangle^{{1/{compensate}}}$", fontsize=fsize)
    ax[1].set_ylabel(rf'Nu(t) / Re(t)$^{{1/{compensate}}}$ ', fontsize=fsize)

ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Mass + Frequency", fontsize=fsize)
plt.tight_layout()

if compensate==0:
    filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/Nu_Re.pdf'
else:
    filename = f'/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/Nu_Re_compensated({compensate}).pdf'
plt.savefig(filename, dpi=200, bbox_inches='tight')

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

fig, ax = plt.subplots(1,3, figsize=(15,5))
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
ax[2].set_ylabel(r'$(T-T_m)/\Delta T$')

plt.tight_layout()

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/Solutions.pdf'
plt.savefig(filename, dpi=400, bbox_inches='tight')

plt.show()

#%%


















