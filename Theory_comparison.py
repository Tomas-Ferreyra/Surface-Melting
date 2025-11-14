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
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.integrate import cumulative_trapezoid

def solution_R( R, A,B ):
    term1 = A*R/B
    prod1 = (A-B) / (6 * np.cbrt(B)**4 * np.cbrt(1+B)**2 )
    acta = 2 * np.sqrt(3) * np.arctan( -(1 + 2* np.cbrt(B/(1+B)) * R ) / np.sqrt(3)  )
    log1 = 2 * np.log( np.cbrt(1+B) - np.cbrt(B)*R )
    log2 = np.log( np.cbrt(1+B)**2 + np.cbrt(1+B) * np.cbrt(B) * R + np.cbrt(B)**2 * R**2 )
    return (term1 + prod1 * (acta + log1 - log2)) 

def solution_T( R, A,B,Tm,To ):
    u = 1 - R**3
    top = (Tm*A + B*(To-Tm)) * u + To
    bot = A*u + 1
    return top/bot


def R_of_T(T, A,B,Tm,To, ):
    bot = A * (Tm-T) + B * (To-Tm)
    argum = 1 - (T-To)/bot 
    return np.cbrt( argum )

def V_of_T(T, A,B,Tm,To):
    bot = A * (Tm-T) + B * (To-Tm)
    argum = 1 - (T-To)/bot 
    return  argum 

def V_eloss_term(t,T, Tm,rhoi,Vo,L,cp,m,b):
    integrand = m * T + b
    bot = rhoi * Vo * (L + cp*(T-Tm)) 
    eloss = cumulative_trapezoid(integrand, t, initial=0)
    return eloss / bot


def constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L ):
    beta = rhoi / rhow
    gamma = Vb/Vo
    A = beta/ gamma
    B = -L*beta / (gamma*cp*(To-Tm))
    C = (To-Tm)/Vo**(1/3)
    
    Ste = L / (cp * (To-Tm))
    return A,B,C, beta,Ste

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

plt.figure(figsize=(14, 8))

for i, label in enumerate(sorted(results.keys(), key=sort_key)):
    folder_name, freq = label.split()
    data = results[label]

    t = data["t_avg"]
    T = data["T_top_avg"]
    T_init = data["T_top_init"]

    # Plot main temperature line
    plt.plot(t, T,
             label=rf"{label} ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
             color=get_color(folder_name, freq),
             linestyle='-',
             marker='o', markersize=8,
             markeredgecolor='black', linewidth=1)

    # --- Identify and plot the minimum point ---
    min_idx = np.argmin(T)
    t_min = t[min_idx]
    T_min = T[min_idx]

    # Plot star marker at minimum
    plt.plot(t_min, T_min, marker='*', color='yellow',
             markersize=14, markeredgecolor='black', linewidth=1, zorder=5, label=None)

    # --- Print the minimum to terminal ---
    print(f"⭐ {label}: Min Temperature = {T_min:.2f}°C at {t_min:.1f} s")

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
cutoffs = {5:0.5, 10:0.61, 20:0.7}

# 5 kg - 4Hz

mass_fig = [5,10,20]
# mass_fig = [20]

cs, freqs, mss, tinis = [],[],[],[]
cerr = []

ratio = 1.5
plt.figure( figsize=(14/ratio,8/ratio) ) #figsize=(14, 8))

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
        
        A,B,C, beta,Ste = constants( To, Tm, Vo, Vb, rhoi, rhow, cp, L )
        
        ctime = np.real( -( solution_R(R, A, B) - solution_R(1, A, B) ) )
        temp = solution_T(R, A, B, Tm, To)
        
        # --- Identify and plot the minimum point ---
        min_idx = np.argmin(T)
        t_min = t[min_idx]
        T_min = T[min_idx]
    
    
        # mask = t<t_min        
        fitfun = lambda Temp,c: np.real( -( solution_R( R_of_T(Temp, A, B, Tm, To) , A, B) - solution_R(1, A, B) ) ) / c
        
        
        # c, _ = curve_fit(fitfun, T[mask], t[mask])
    
        # # Plot main temperature line
        # plt.plot(t, T,
        #          label=rf"{label} ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
        #          color=get_color(folder_name, freq),
        #          linestyle='-',
        #          marker='o', markersize=8,
        #          markeredgecolor='black', linewidth=1)
    
        # # Plot star marker at minimum
        # plt.plot(t_min, T_min, marker='*', color='yellow',
        #           markersize=14, markeredgecolor='black', linewidth=1, zorder=5, label=None)
        
            
        # Plot main temperature line using function fitfun (should be linear some part)
        T_linear = fitfun(T,1)
        mask = T_linear < cutoffs[mass]
        constant_sl = lambda T,c: c*T
        c,cov = curve_fit(constant_sl, t[mask], T_linear[mask] )
        c = c[0]
        print(label,'kg, c = {:.4f}'.format(c) )
        print('\t No ice:{:.3f}, {:.1f}% of ice: {:.3f}'.format(solution_R(1, A, B) - solution_R(0, A, B), 0.55**3 * 100 ,solution_R(1, A, B) - solution_R(0.55, A, B)))
        # print('\t 7°C:{:.3f}, 12°C: {:.3f}'.format(solution_R(1, A, B) - solution_R(R_of_T( 7, A, B, Tm, To) , A, B),  \
        #                                            solution_R(1, A, B) - solution_R(R_of_T(12, A, B, Tm, To), A, B)))

        frequ = int(re.split('-| ',label)[-1][:-2])
        cs.append( c )
        freqs.append( frequ )
        mss.append( mass )
        tinis.append(T_init)
        cerr.append( np.sqrt(cov[0,0]) )
    
        plt.plot(t, T_linear,
                  # label=rf"{mass}kg, {freq} ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                  color=get_color(folder_name, freq),
                  linestyle='-',
                  marker='o', markersize=8,
                  markeredgecolor='black', linewidth=1)
        
        plt.plot( t[mask], c*t[mask], 'r--' )

        # plt.plot(t*c, T_linear,
        #           # label=rf"{mass}kg, {freq} ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
        #           color=get_color(folder_name, freq),
        #           linestyle='-',
        #           marker='o', markersize=8,
        #           markeredgecolor='black', linewidth=1)
        
        # plt.plot(c*t, T,
        #          # label=rf"{mass}kg, {freq} ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
        #          color=get_color(folder_name, freq),
        #          linestyle='-',
        #          marker='o', markersize=8,
        #          markeredgecolor='black', linewidth=1)
        
        # plt.plot( t[mask], c*t[mask], 'r--' )
        
    # Plot star marker at minimum
    # plt.plot(t_min, fitfun(T_min,1), marker='*', color='yellow',
    #           markersize=14, markeredgecolor='black', linewidth=1, zorder=5, label=None)

    
    # plt.plot(ctime / c , temp, 'r--')

    # # --- Print the minimum to terminal ---
    # print(f"⭐ {label}: Min Temperature = {T_min:.2f}°C at {t_min:.1f} s", c)

# plt.hlines(cutoffs[mass_fig[0]], 0, 500, color='black', linestyles='dashed', label='Fit limit' ) #plot limit of fit 
# plt.plot([0,1],[0,1],'--',color='orange')

fsize = 12
plt.xlabel(r"$Ct$ ", fontsize=fsize )
plt.ylabel(r"$R_s(R_T(T)) - R_s(1)$", fontsize=fsize )
# plt.title(f"{mass_fig[0]} kg", fontsize=fsize )
plt.grid(True)

# plt.xscale('log')
# plt.yscale('log')

# filename = f'/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/model_fit_{mass_fig[0]}kg.pdf'
filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/model_all_mass_fitted.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')
print()
print(filename)

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Folder + Frequency")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# plt.figure()
# # plt.plot(ctime / 0.001, R, '.-')
# plt.plot(ctime / c , temp, '-')
# plt.show()

#%%
cs = np.array(cs)
cerr = np.array(cerr)
mss = np.array(mss)
freqs = np.array(freqs) 
tinis = np.array(tinis) 

umrss = np.array([0.003957, 0.00817, 0.01844, 0.03881, 0.06663])
color = [0,'blue','green',0,'red']
# rhow/rhoi*cp/L * umrss

fig, ax = plt.subplots(1,2, figsize=(10,4), layout="constrained")
for i in [5,10,20]:
    mask = mss==i
    # ax[0].plot( freqs[mask], cs[mask] , 'o', label=f'{i} kg', color=color[i//5], markeredgecolor='k')
    # ax[1].plot( freqs[mask], cs[mask] * np.cbrt(i / rhoi) / tinis[mask]  , 'o', label=f'{i} kg', color=color[i//5], markeredgecolor='k')

    ax[0].errorbar( freqs[mask], cs[mask], yerr=cerr[mask] , fmt='o', label=f'{i} kg', color=color[i//5], markeredgecolor='k')
    ax[1].errorbar( freqs[mask], cs[mask] * np.cbrt(i / rhoi) / tinis[mask], yerr=cerr[mask] * np.cbrt(i / rhoi) / tinis[mask] , \
                  fmt='o', label=f'{i} kg', color=color[i//5], markeredgecolor='k')

ax[1].plot( [1,2,4,8,12], rhow/rhoi*cp/L * umrss * 0.09  , '^', color='orange', label='theory?',zorder=4)

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

#Plot Radius and Volume over time (or its derivatives over time)
fig, ax = plt.subplots(1,2,figsize=(18,8))

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
        mask2 = Vsv<0
        if np.sum(mask2)>0:
            fin = np.min( [np.where(mask1)[0][0], np.where(mask2)[0][0] ]) 
        else:
            fin = np.where(mask1)[0][0]
        

    # Plot radius and volume over time (with this energy balance)
    if not gradient:
        ax[1].plot(t[:fin], R[:fin],
                  label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                  color=get_color(folder_name, freq),
                  linestyle='-',
                  marker='o', markersize=8,
                  markeredgecolor='black', linewidth=1)

        if show_minima: ax[1].plot( t[fin], R[fin], marker='*', color='yellow',
                  markersize=14, markeredgecolor='black', linewidth=1, zorder=5, label=None )
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

fsize = 15
ax[0].tick_params(axis='both', which='major', labelsize=fsize)
ax[1].tick_params(axis='both', which='major', labelsize=fsize)
ax[0].set_xlabel("t (s)", fontsize=fsize)
ax[1].set_xlabel("t (s)", fontsize=fsize)
if not gradient:
    ax[0].set_ylabel(r"$\tilde{V}$ ", fontsize=fsize)
    ax[1].set_ylabel(r"$\tilde{R}$ ", fontsize=fsize)
else:
    ax[1].set_ylabel(r"$d\tilde{V}/dt$ ", fontsize=fsize)
    ax[0].set_ylabel(r"$d\tilde{R}/dt$ ", fontsize=fsize)
ax[1].set_title("Radius vs Time", fontsize=fsize)
ax[0].set_title("Volume vs Time", fontsize=fsize)

ax[0].grid(True)
ax[1].grid(True)
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Mass + Frequency", fontsize=fsize)
plt.tight_layout()

filename = '/Volumes/ICESTOCKS/Ice Stocks/new_transfer_tolga/Figures/temperatures.pdf'
# plt.savefig(filename, dpi=200, bbox_inches='tight')

plt.show()

#%%

apply_heat_loss = False

# dR/dt vs Temperature
fig, ax = plt.subplots(1,2,figsize=(18,8))

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

compensate = 1

umrss = {1:0.003957, 2:0.00817, 4:0.01844, 8:0.03881, 12:0.06663} #m/s, u_rms
masses_bath = {5:117, 10:112, 20:102} #in kg

# fig, ax = plt.subplots(1,2,figsize=(14,8) )
fig, ax = plt.subplots(1,2,figsize=(18,8), sharey=False )

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
    mask2 = Vsv<0
    if np.sum(mask2)>0:
        fin = np.min( [np.where(mask1)[0][0], np.where(mask2)[0][0] ]) 
    else:
        fin = np.where(mask1)[0][0]

    Tsv = savgol_filter(T, 40, 3)
    Rsv = np.cbrt(Vsv)
    gRsv = np.gradient(Rsv,t)
    
    Nu = - beta * Ste * Pr * np.mean(gRsv[:fin]) * Rsv[0] * np.cbrt(Vo)**2 / nu
    Re = u_rms * Rsv[0] * np.cbrt(Vo) / nu
    
    Nut = - beta * Ste * Pr * gRsv[:fin] * Rsv[:fin] * np.cbrt(Vo)**2 / nu
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
        ax[1].scatter(Ret, Nut / Re**(1/compensate) , 
                      label=rf"{mass} kg, {frequency} Hz, ($T_{{\mathrm{{init}}}}$: {T_init:.2f}°C)",
                      color=get_color(folder_name, freq),
                      linestyle='-',
                      marker='o', s=30,
                      edgecolors='black', linewidth=1)
    
if compensate == 0:
    # res = np.logspace(3.3,4.3)
    # ax[0].plot( res, res*0.3, 'k--', label=r'Nu $\propto$ Re' )  
    res = np.logspace(2.5,4.1)
    ax[1].plot( res, res, 'k--', label=r'Nu $\propto$ Re' )  
    ax[1].plot( res, res**(1/2) * 30, 'm--', label=r'Nu $\propto$ Re$^{1/2}$' )  
else:
    # res = np.logspace(3.3,4.3)
    # ax[0].plot( res, res*0.3, 'k--', label=r'Nu $\propto$ Re' )  
    res = np.logspace(2.5,4.1)
    ax[1].plot( res, res / res**(1/compensate), 'k--', label=r'Nu $\propto$ Re' )  
    ax[1].plot( res, res**(1/2) * 1.5 / res**(1/compensate), 'm--', label=r'Nu $\propto$ Re$^{1/2}$' )  
    
ax[0].set_xscale('log')    
ax[0].set_yscale('log')    
ax[0].set_xlabel(r'$\langle$Re$\rangle$')

    
ax[1].set_xscale('log')    
ax[1].set_yscale('log')    
ax[1].set_xlabel('Re(t)')

if compensate == 0:
    ax[0].set_ylabel(r"$\langle$Nu$\rangle$")
    ax[1].set_ylabel(r'Nu(t)')
else:
    ax[0].set_ylabel(rf"$\langle$Nu$\rangle$ / $\langle$Re$\rangle^{{1/{compensate}}}$")
    ax[1].set_ylabel(rf'Nu(t) / Re(t)$^{{1/{compensate}}}$ ')

ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Mass + Frequency", fontsize=fsize)
plt.tight_layout()
plt.show()
#%%




#%%



#%%

