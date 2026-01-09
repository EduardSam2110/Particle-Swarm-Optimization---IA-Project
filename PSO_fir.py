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


# --- INTERFATA GRAFICA (GUI) ---

class PSOApp:
    def __init__(self, master):
        self.master = master
        master.title("PSO pentru Design Filtru FIR (Configurabil)")

        # Parametri
        self.params = {
            'n_coeff': tk.IntVar(value=10),
            'w': tk.DoubleVar(value=0.729),
            'c1': tk.DoubleVar(value=1.5),
            'c2': tk.DoubleVar(value=1.5),
            'swarm_size': tk.IntVar(value=30),
            'max_iter': tk.IntVar(value=100),
            'mode': tk.IntVar(value=GLOBAL),
            'filter_type': tk.StringVar(value="Trece-Jos"),
            'f_cutoff1': tk.DoubleVar(value=0.4),  # Frecventa 1 in termeni de pi (ex: 0.4 -> 0.4*pi)
            'f_cutoff2': tk.DoubleVar(value=0.6),  # Frecventa 2 in termeni de pi
            'l': tk.IntVar(value=1)
        }

        self.setup_input_frame()
        self.setup_plot_frame()

    def toggle_l_field(self):
        if self.params['mode'].get() == LOCAL:
            self.l_entry.config(state='normal')
        else:
            self.l_entry.config(state='disabled')

    def setup_input_frame(self):
        frame = ttk.Frame(self.master, padding="10")
        frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(frame, text="Specificatii Filtru", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                                                      pady=5)

        # 1. Tip Filtru
        ttk.Label(frame, text="Tip Filtru").grid(row=1, column=0, sticky=tk.W)
        self.filter_type_combo = ttk.Combobox(frame, textvariable=self.params['filter_type'], width=17,
                                              state="readonly",
                                              values=["Trece-Jos", "Trece-Sus", "Trece-Banda", "Opreste-Banda"])
        self.filter_type_combo.grid(row=1, column=1, pady=2)
        self.filter_type_combo.bind("<<ComboboxSelected>>", self.update_cutoff_fields)

        # 2. Frecventa de taiere 1
        self.label_f1 = ttk.Label(frame, text=r"Frecv. 1 (ω/π)")
        self.label_f1.grid(row=2, column=0, sticky=tk.W)
        self.entry_f1 = ttk.Entry(frame, textvariable=self.params['f_cutoff1'], width=10)
        self.entry_f1.grid(row=2, column=1, pady=2)

        # 3. Frecventa de taiere 2
        self.label_f2 = ttk.Label(frame, text=r"Frecv. 2 (ω/π)")
        self.label_f2.grid(row=3, column=0, sticky=tk.W)
        self.entry_f2 = ttk.Entry(frame, textvariable=self.params['f_cutoff2'], width=10)
        self.entry_f2.grid(row=3, column=1, pady=2)

        # 4. Numar Coeficienti
        ttk.Label(frame, text="Nr. Coeficienți (D)").grid(row=4, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['n_coeff'], width=10).grid(row=4, column=1, pady=2)

        ttk.Separator(frame, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky='ew', pady=5)

        # Parametri PSO (w, c1, c2, etc.)
        row_start = 6
        ttk.Label(frame, text="w (Inerție)").grid(row=row_start, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['w'], width=10).grid(row=row_start, column=1, pady=2)

        ttk.Label(frame, text="c1 (Cognitiv)").grid(row=row_start + 1, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['c1'], width=10).grid(row=row_start + 1, column=1, pady=2)

        ttk.Label(frame, text="c2 (Social)").grid(row=row_start + 2, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['c2'], width=10).grid(row=row_start + 2, column=1, pady=2)

        ttk.Label(frame, text="Dim. Roi").grid(row=row_start + 3, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['swarm_size'], width=10).grid(row=row_start + 3, column=1, pady=2)

        ttk.Label(frame, text="Max Iter.").grid(row=row_start + 4, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['max_iter'], width=10).grid(row=row_start + 4, column=1, pady=2)

        ttk.Label(frame, text="Topologie").grid(row=row_start + 5, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(frame, text="Global (gbest)", variable=self.params['mode'], value=GLOBAL,command=self.toggle_l_field).grid(
            row=row_start + 6, column=0, sticky=tk.W)
        ttk.Radiobutton(frame, text="Local (lbest)", variable=self.params['mode'], value=LOCAL,command=self.toggle_l_field).grid(row=row_start + 6,
                                                                                                     column=1,
                                                                                                     sticky=tk.W)

        ttk.Label(frame, text="l (neighborhood size) =").grid(row=row_start + 7, column=0, sticky=tk.W)
        self.l_entry=ttk.Entry(frame, textvariable=self.params['l'])
        self.l_entry.grid(row=row_start + 7, column=1)
        self.l_entry.config(state='disabled')

        ttk.Button(frame, text="Rulează Optimizarea", command=self.run_pso_fir).grid(row=row_start + 9, column=0,
                                                                                     columnspan=2, pady=10)

        # Initializare starea campurilor de frecventa
        self.update_cutoff_fields()

    def update_cutoff_fields(self, event=None):
        """Activeaza/dezactiveaza campurile de frecventa in functie de tipul filtrului."""
        filter_type = self.params['filter_type'].get()

        if filter_type in ["Trece-Banda", "Opreste-Banda"]:
            self.entry_f1.config(state='normal')
            self.entry_f2.config(state='normal')
        elif filter_type in ["Trece-Jos", "Trece-Sus"]:
            self.entry_f1.config(state='normal')
            self.entry_f2.config(state='disabled')
        else:
            self.entry_f1.config(state='disabled')
            self.entry_f2.config(state='disabled')

    def setup_plot_frame(self):
        self.plot_frame = ttk.Frame(self.master)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=1)

        self.ax.set_title("Raspunsul in Frecventa al Filtrului FIR")
        self.ax.set_xlabel("Frecvență ω (rad/sample)")
        self.ax.set_ylabel("Magnitudine |H(ω)|")
        self.ax.grid(True)

        self.response_line, = self.ax.plot([], [], label='Răspuns PSO', color='blue')
        self.ideal_line, = self.ax.plot([], [], label='Răspuns Ideal', color='red', linestyle='--')
        self.ax.legend()

        self.mse_text = self.ax.text(0.05, 0.95, "MSE: N/A", transform=self.ax.transAxes,
                                     fontsize=10, verticalalignment='top',
                                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

    def run_pso_fir(self):
        global FILTER_TYPE_GLOBAL, CUTOFFS_GLOBAL

        # 1. Validare si extragere parametri
        try:
            n_coeff = self.params['n_coeff'].get()
            w = self.params['w'].get()
            c1 = self.params['c1'].get()
            c2 = self.params['c2'].get()
            swarm_size = self.params['swarm_size'].get()
            max_iter = self.params['max_iter'].get()
            mode = self.params['mode'].get()

            filter_type = self.params['filter_type'].get()
            f1 = self.params['f_cutoff1'].get()
            f2 = self.params['f_cutoff2'].get()

            # Validare Filtru
            if f1 < 0 or f1 > 1.0 or (f2 < 0 or f2 > 1.0):
                raise ValueError("Frecvențele trebuie să fie între 0 și 1 (corespunde la 0 și π rad/sample).")
            if n_coeff <= 0:
                raise ValueError("Numărul de coeficienți trebuie să fie pozitiv.")

            # Setare variabile globale pentru functia fitness
            FILTER_TYPE_GLOBAL = filter_type
            if filter_type in ["Trece-Banda", "Opreste-Banda"]:
                CUTOFFS_GLOBAL = [f1, f2]
            else:
                CUTOFFS_GLOBAL = [f1]

        except Exception as e:
            messagebox.showerror("Eroare la parametri", f"Verificați formatul parametrilor introduși. Eroare: {e}")
            return

        # 2. Rulare PSO
        self.mse_text.set_text("Rulare PSO...")
        self.master.update()

        try:
            swarm = PSO(mode, swarm_size, w, c1, c2, max_iter, n_coeff)
        except Exception as e:
            messagebox.showerror("Eroare la rulare", f"Eroare în logica PSO: {e}")
            return

        # 3. Analiza rezultatului
        best_particle = min(swarm, key=lambda p: fitness(p.y))
        best_coeffs = best_particle.y
        final_mse = fitness(best_coeffs)

        # 4. Vizualizare si raportare
        self.plot_fir_response(best_coeffs, final_mse, n_coeff, mode, filter_type, CUTOFFS_GLOBAL)
        messagebox.showinfo("Finalizat", f"Optimizarea a fost finalizată.\nMSE final: {final_mse:.6f}")

    def plot_fir_response(self, h_coeffs, final_mse, n_coeff, mode, filter_type, cutoffs):

        # 1. Calculeaza Raspunsul Real si Ideal
        w, H_real = freqz(h_coeffs, worN=1024)

        H_ideal_vals = H_ideal(w)

        # 2. Sterge si Re-deseneaza graficul
        self.ax.clear()

        # Raspuns Real (PSO)
        self.response_line, = self.ax.plot(w, np.abs(H_real), label=f'Răspuns PSO (h={n_coeff})', color='blue',
                                           linewidth=2)

        # Raspuns Ideal
        self.ideal_line, = self.ax.plot(w, H_ideal_vals, label=f'Ideal ({filter_type})', color='red', linestyle='--')

        # 3. Setari Grafic
        mode_str = "GBEST" if mode == GLOBAL else "LBEST"
        if len(cutoffs) == 1:
            f_str = f"ωc={cutoffs[0]:.2f}π"
        else:
            f_str = f"ω1={cutoffs[0]:.2f}π, ω2={cutoffs[1]:.2f}π"

        self.ax.set_title(f"Design Filtru FIR ({mode_str}): {filter_type} cu {f_str}")
        self.ax.set_xlabel("Frecvență ω (rad/sample)")
        self.ax.set_ylabel("Magnitudine |H(ω)|")
        self.ax.set_ylim(-0.1, 1.2)
        self.ax.set_xlim(0, np.pi)  # Frecventa este de la 0 la pi
        self.ax.grid(True)
        self.ax.legend()

        # 4. Actualizare Text MSE
        mse_text_content = f"MSE Final: {final_mse:.6f}\nCoeficienți (primii 5): {h_coeffs[:5].round(4)}"
        self.mse_text = self.ax.text(0.05, 0.95, mse_text_content, transform=self.ax.transAxes,
                                     fontsize=10, verticalalignment='top',
                                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

        self.canvas.draw()

def on_closing():
    root.destroy()  
    sys.exit()  

if __name__ == "__main__":
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = PSOApp(root)
    root.mainloop()