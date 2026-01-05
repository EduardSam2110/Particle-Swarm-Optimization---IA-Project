import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from random import Random
import math

GLOBAL = 1
LOCAL = 0

class Particle:
    def __init__(self):
        self.x = 0 #current position of the particle
        self.v = 0 #velocity
        self.y = 0 #personal best position (pbest)
        self.yn = 0 #neighborhood best position (lbest sau gbest)


USER_FITNESS_FUNC = None
SEARCH_BOUNDS = (-5, 5) #interval


def fitness(x):
    """Evalueaza functia de fitness definita de utilizator."""
    global USER_FITNESS_FUNC
    if USER_FITNESS_FUNC is None or USER_FITNESS_FUNC == "":
        return x ** 4 + 16  #Functie predefinita

    try:
        result = eval(USER_FITNESS_FUNC, {'x': x, 'math': math, 'np': np})
        return result
    except Exception:
        return float('inf')


def calc_personal_best(p: Particle):
    """Calculeaza si actualizeaza pbest."""
    if fitness(p.x) < fitness(p.y):
        return p.x
    else:
        return p.y


def calc_neighborhood_best(swarm):
    """Calculeaza gbest."""
    if not swarm: return 0
    best_y = swarm[0].y
    for p in swarm:
        if fitness(p.y) < fitness(best_y):
            best_y = p.y
    return best_y


def calc_local_best(swarm, i,l=1):
    """Calculeaza (lbest) pentru particula i (default avem topologie inelara k=3)."""
    n = len(swarm)
    neighbors = [swarm[(i + offset) % n] for offset in range(-l, l + 1)]
    best_y = neighbors[0].y
    for p in neighbors:
        if fitness(p.y) < fitness(best_y):
            best_y = p.y
    return best_y


def PSO(mode, swarm_size, w, c1, c2, max_iter, l=1):
    """Functia principala a algoritmului PSO."""
    rand = Random()
    swarm = [Particle() for _ in range(swarm_size)]
    history = []

    x_min, x_max = SEARCH_BOUNDS

    for p in swarm:
        p.x = rand.uniform(x_min, x_max)
        p.v = 0
        p.y = p.x

    for _ in range(max_iter):

        # 1. Actualizare pbest (y)
        for i, p in enumerate(swarm):
            p.y = calc_personal_best(p)

        # 2. Calculul celei mai bune pozitii sociale (yn - gbest sau lbest)
        if mode == GLOBAL:
            yn = calc_neighborhood_best(swarm)
            for p in swarm:
                p.yn = yn
        else:
            for i, p in enumerate(swarm):
                p.yn = calc_local_best(swarm, i, l)

                # 3. Inregistrare istoric
        history.append([p.x for p in swarm])

        # 4. Actualizare viteza (v) si pozitie (x)
        for p in swarm:
            r1, r2 = rand.random(), rand.random()

            p.v = (w * p.v +
                   c1 * r1 * (p.y - p.x) +
                   c2 * r2 * (p.yn - p.x))

            p.x += p.v

            # Asigurare in limite
            p.x = np.clip(p.x, x_min, x_max)

    return swarm, history


# --- INTERFATA GRAFICA (GUI) ---

class PSOApp:
    def __init__(self, master):
        self.master = master
        master.title("PSO gasire minim functie")

        # Parametri impliciti
        self.params = {
            'w': tk.DoubleVar(value=0.5),
            'c1': tk.DoubleVar(value=1.5),
            'c2': tk.DoubleVar(value=1.5),
            'swarm_size': tk.IntVar(value=10),
            'max_iter': tk.IntVar(value=50),
            'mode': tk.IntVar(value=GLOBAL),
            'fitness_str': tk.StringVar(value='x**4 + 16'),
            'x_min': tk.DoubleVar(value=-5.0),
            'x_max': tk.DoubleVar(value=5.0),
            'l': tk.IntVar(value=1)
        }

        self.result_text_obj = None  # Obiectul de text pe care il vom actualiza

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

        ttk.Label(frame, text="Parametri PSO", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=5)

        ttk.Label(frame, text="f(x) (ex: x**2 + 1)").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['fitness_str'], width=20).grid(row=1, column=1, pady=2)

        ttk.Label(frame, text="w (Inertie)").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['w'], width=10).grid(row=2, column=1, pady=2)

        ttk.Label(frame, text="c1 (Cognitiv)").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['c1'], width=10).grid(row=3, column=1, pady=2)

        ttk.Label(frame, text="c2 (Social)").grid(row=4, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['c2'], width=10).grid(row=4, column=1, pady=2)

        ttk.Label(frame, text="Dim. Roi").grid(row=5, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['swarm_size'], width=10).grid(row=5, column=1, pady=2)

        ttk.Label(frame, text="Max Iter.").grid(row=6, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['max_iter'], width=10).grid(row=6, column=1, pady=2)

        ttk.Label(frame, text="Interval").grid(row=7, column=0, columnspan=2, pady=(10, 0))

        ttk.Label(frame, text="x min").grid(row=8, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['x_min']).grid(row=8, column=1)

        ttk.Label(frame, text="x max").grid(row=9, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.params['x_max']).grid(row=9, column=1)

        ttk.Label(frame, text="Topologie").grid(row=10, column=0, columnspan=2, pady=10)
        ttk.Radiobutton(frame, text="Global (gbest)", variable=self.params['mode'], value=GLOBAL,command=self.toggle_l_field).grid(row=11, column=0,
                                                                                                       sticky=tk.W)
        ttk.Radiobutton(frame, text="Local (lbest)", variable=self.params['mode'], value=LOCAL,command=self.toggle_l_field).grid(row=11, column=1,
                                                                                                     sticky=tk.W)
        ttk.Label(frame, text="l (neighborhood size) =").grid(row=12, column=0, sticky=tk.W)
        self.l_entry=ttk.Entry(frame, textvariable=self.params['l'])
        self.l_entry.grid(row=12, column=1)
        self.l_entry.config(state='disabled')

        ttk.Button(frame, text="Ruleaza PSO & Animeaza", command=self.run_pso).grid(row=13, column=0, columnspan=2,
                                                                                    pady=13)

    def setup_plot_frame(self):
        self.plot_frame = ttk.Frame(self.master)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=1)

        self.ax.set_title("Rezultat PSO (Asteptare)")
        self.ax.grid(True)

        # Initializare linii de plotare (vor fi actualizate)
        self.line_func, = self.ax.plot([], [], label='f(x)', color='gray', linestyle='--')
        self.scatter_particles = self.ax.scatter([], [], c='yellow', s=50, edgecolors='black', label='Particule')
        self.ax.legend()

        # Initializare obiect text care va afisa rezultatul final
        self.result_text_obj = self.ax.text(0.05, 0.95, "Asteapta rularea...", transform=self.ax.transAxes,
                                            fontsize=10, verticalalignment='top',
                                            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

    def run_pso(self):
        global USER_FITNESS_FUNC, SEARCH_BOUNDS

        # 1. Validare si extragere parametri
        try:
            w = self.params['w'].get()
            c1 = self.params['c1'].get()
            c2 = self.params['c2'].get()
            swarm_size = self.params['swarm_size'].get()
            max_iter = self.params['max_iter'].get()
            mode = self.params['mode'].get()
            USER_FITNESS_FUNC = self.params['fitness_str'].get()
            x_min = self.params['x_min'].get()
            x_max = self.params['x_max'].get()
            l = self.params['l'].get()

            if x_min >= x_max:
                messagebox.showerror("Eroare", "x_min trebuie sa fie mai mic decat x_max.")
                return

            SEARCH_BOUNDS = (x_min, x_max)

            # Test de validare a functiei
            test_val = fitness(1.0)
            if test_val == float('inf'):
                messagebox.showerror("Eroare",
                                     "Functia de fitness introdusa nu este valida. Verifica sintaxa (ex: 'x**4 + 16').")
                return

        except Exception as e:
            messagebox.showerror("Eroare", f"Verificati formatul parametrilor introdusi. Eroare: {e}")
            return

        # 2. Rulare PSO
        try:
            swarm, history = PSO(mode, swarm_size, w, c1, c2, max_iter,l)
        except Exception as e:
            messagebox.showerror("Eroare la rulare", f"Eroare in logica PSO: {e}")
            return

        # 3. Gaseste cel mai bun rezultat final
        best_particle = min(swarm, key=lambda p: fitness(p.y))

        # 4. Pornire Animatie
        self.animate_results(history, max_iter, mode, best_particle)

    def update_result_text(self, final_best):
        """Actualizeaza doar textul rezultatului pe grafic."""
        final_x = final_best.y
        final_f = fitness(final_x)

        # Calculeaza textul
        result_text = f"Minim Găsit:\nx = {final_x:.4f}\nf(x) = {final_f:.4f}"

        # Actualizeaza textul existent
        self.result_text_obj.set_text(result_text)

    def animate_results(self, history, max_iter, mode, final_best):

        # 1. Plotare Curba Functiei (Re-desenare)
        x_min, x_max = SEARCH_BOUNDS
        xs_plot = np.linspace(x_min, x_max, 200)
        ys_plot = np.array([fitness(x) for x in xs_plot])

        # Determina limitele y pentru vizualizare
        y_min_plot = np.min(ys_plot)
        y_max_plot = np.max(ys_plot)
        y_range = y_max_plot - y_min_plot

        self.ax.clear()

        # Re-initializeaza linia functiei si particulele
        self.line_func, = self.ax.plot(xs_plot, ys_plot, label='f(x)', color='gray', linestyle='--')

        # Plotarea minimului gasit (final)
        self.ax.plot(final_best.y, fitness(final_best.y), 'gx', markersize=15, markeredgewidth=2, label='Minim PSO')

        # 2. Re-initializare Scatter (Particule)
        initial_x = history[0]
        initial_y = [fitness(x) for x in initial_x]
        self.scatter_particles = self.ax.scatter(initial_x, initial_y, c='yellow', s=50, edgecolors='black',
                                                 label='Particule')

        # 3. Re-creare obiect text (pentru a evita eroarea, trebuie recreat dupa ax.clear())
        self.result_text_obj = self.ax.text(0.05, 0.95, "Asteapta rularea...", transform=self.ax.transAxes,
                                            fontsize=10, verticalalignment='top',
                                            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

        # 4. Setare Axa si Titlu
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min_plot - 0.1 * y_range, y_max_plot + 0.1 * y_range)
        self.ax.set_xlabel('Pozitia X')
        self.ax.set_ylabel('Valoarea Functiei f(x)')
        self.ax.grid(True)
        self.ax.legend()

        # 5. Actualizeaza textul cu rezultatul initial
        self.update_result_text(final_best)

        # Functia de actualizare (Animation Step)
        def update_plot(frame):
            if frame < len(history):
                current_x = history[frame]
                current_y = [fitness(x) for x in current_x]

                # Actualizeaza pozitiile particulelor
                self.scatter_particles.set_offsets(np.c_[current_x, current_y])

                # Actualizeaza titlul cu numarul iteratiei
                self.ax.set_title(f'PSO ({"GBEST" if mode == GLOBAL else "LBEST"}) - Iteratia {frame + 1}/{max_iter}')

            return self.scatter_particles,

        # 6. Ruleaza animatia Tkinter
        self.master.after(0, lambda: self._run_animation_loop(history, max_iter, update_plot, 0))

    def _run_animation_loop(self, history, max_iter, update_plot, frame):
        if frame < max_iter:
            update_plot(frame)
            self.canvas.draw_idle()
            # Ruleaza urmatoarea iteratie dupa 100ms
            self.master.after(100, lambda: self._run_animation_loop(history, max_iter, update_plot, frame + 1))
        else:
            self.ax.set_title(
                f'PSO ({"GBEST" if self.params["mode"].get() == GLOBAL else "LBEST"}) - Optimizare Finalizata')
            self.canvas.draw_idle()
            
            #messagebox.showinfo("Finalizat", "Optimizarea a fost finalizata.")


if __name__ == "__main__":
    root = tk.Tk()
    app = PSOApp(root)
    root.mainloop()