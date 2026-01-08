import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from random import Random
from scipy.signal import freqz
import sys

GLOBAL = 1
LOCAL = 0


class Particle:
    def __init__(self):
        self.x = 0
        self.v = 0
        self.y = 0  # pbest
        self.yn = 0 # lbest/gbest


# Parametrii problemei FIR
N_COEFF_GUI = 10  # Dimensiunea spatiului (va fi actualizata din GUI)
SEARCH_BOUNDS = (-1.0, 1.0)  # Limitele de cautare pentru coeficienti

# Variabile globale pentru specificatiile filtrului
FILTER_TYPE_GLOBAL = "Trece-Jos"
CUTOFFS_GLOBAL = [0.4]  # Frecvente de taiere (normalized angular frequency / pi)


def H_ideal(w):
    """
    Raspunsul ideal al filtrului bazat pe tip si frecvente.
    Frecventa w este in [0, pi].
    """
    global FILTER_TYPE_GLOBAL, CUTOFFS_GLOBAL

    w_norm = w / np.pi  # Normalizeaza frecventa la [0, 1]

    H = np.zeros_like(w)

    if FILTER_TYPE_GLOBAL in ["Trece-Jos", "Trece-Sus"]:
        fc = CUTOFFS_GLOBAL[0] if CUTOFFS_GLOBAL else 0.4
        fc = np.clip(fc, 0, 1)
    elif len(CUTOFFS_GLOBAL) == 2:
        fc1, fc2 = min(CUTOFFS_GLOBAL), max(CUTOFFS_GLOBAL)
        fc1 = np.clip(fc1, 0, 1)
        fc2 = np.clip(fc2, 0, 1)
    else:
        return np.where(w_norm <= 0.4, 1.0, 0.0)

    if FILTER_TYPE_GLOBAL == "Trece-Jos":
        H = np.where(w_norm <= fc, 1.0, 0.0)
    elif FILTER_TYPE_GLOBAL == "Trece-Sus":
        H = np.where(w_norm >= fc, 1.0, 0.0)
    elif FILTER_TYPE_GLOBAL == "Trece-Banda":
        H = np.where((w_norm >= fc1) & (w_norm <= fc2), 1.0, 0.0)
    elif FILTER_TYPE_GLOBAL == "Opreste-Banda":
        H = np.where((w_norm < fc1) | (w_norm > fc2), 1.0, 0.0)

    return H


def fitness(h_coeffs):
    """
    Functia de fitness: Eroarea Patratica Medie (MSE) intre raspunsul real si cel ideal.
    """
    global N_COEFF_GUI
    if len(h_coeffs) != N_COEFF_GUI:
        h_coeffs = h_coeffs[:N_COEFF_GUI]

    w, H_real = freqz(h_coeffs, worN=1024)
    H_ideal_vals = H_ideal(w)

    error = np.mean((np.abs(H_real) - H_ideal_vals) ** 2)

    return error


# --- LOGICA PSO (Adaptata pentru Vectori) ---

def calc_personal_best(p: Particle):
    if fitness(p.x) < fitness(p.y):
        return p.x.copy()
    else:
        return p.y.copy()


def calc_neighborhood_best(swarm):
    if not swarm: return np.zeros(N_COEFF_GUI)
    best_particle = min(swarm, key=lambda p: fitness(p.y))
    return best_particle.y.copy()


def calc_local_best(swarm, i,l=1):
    n = len(swarm)
    #neighbors = [swarm[(i - 1) % n], swarm[i], swarm[(i + 1) % n]]
    neighbors = [swarm[(i + offset) % n] for offset in range(-l, l + 1)]
    best_particle = min(neighbors, key=lambda p: fitness(p.y))
    return best_particle.y.copy()


def PSO(mode, swarm_size, w, c1, c2, max_iter, n_coeff):
    """Functia principala a algoritmului PSO."""
    global N_COEFF_GUI
    N_COEFF_GUI = n_coeff

    rand = Random()
    swarm = [Particle() for _ in range(swarm_size)]

    x_min, x_max = SEARCH_BOUNDS

    w = float(w);
    c1 = float(c1);
    c2 = float(c2)

    for p in swarm:
        p.x = np.array([rand.uniform(x_min, x_max) for _ in range(n_coeff)])
        p.v = np.zeros(n_coeff)
        p.y = p.x.copy()

    for _ in range(max_iter):

        for i, p in enumerate(swarm):
            p.y = calc_personal_best(p)

        if mode == GLOBAL:
            yn = calc_neighborhood_best(swarm)
            for p in swarm:
                p.yn = yn
        else:
            for i, p in enumerate(swarm):
                p.yn = calc_local_best(swarm, i)

        for p in swarm:
            r1 = np.array([rand.random() for _ in range(n_coeff)])
            r2 = np.array([rand.random() for _ in range(n_coeff)])

            p.v = (w * p.v + c1 * r1 * (p.y - p.x) + c2 * r2 * (p.yn - p.x))
            p.x += p.v
            p.x = np.clip(p.x, x_min, x_max)

    return swarm

