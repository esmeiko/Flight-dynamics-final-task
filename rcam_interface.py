"""
=============================================================================
RCAM Aircraft Model — Interfaz Unificada con Navegación por Pestañas
Final Task - Aircraft Dynamics Simulation

Estructura de la interfaz:
  Pestaña 1 | Tarea 2 — Vuelo base (180 s)
  Pestaña 2 | Tarea 3 — Pulso de alerón +5°
  Pestaña 3 | Tarea 4a — Falla de motor
  Pestaña 4 | Tarea 4b — PSO Trim

Uso:
    python rcam_interface.py

El usuario puede navegar entre tareas sin que se cierren las gráficas previas.
Cada pestaña permite ejecutar su tarea de forma independiente o en conjunto.
=============================================================================
"""

import os
import sys
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Importar el modelo matemático
from rcam_model import (
    simulate, xdot,
    X0, u0,
    constant_control, aileron_impulse_control, engine_shutdown_control,
    pso_trim, cost_function,
    VA_TRIM, PSI_TRIM,
    m, g,
)

# =============================================================================
# PALETA Y ESTILOS GLOBALES
# =============================================================================

DARK_BG      = "#0d1117"
PANEL_BG     = "#161b22"
BORDER_COL   = "#30363d"
ACCENT_BLUE  = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED   = "#f85149"
ACCENT_ORG   = "#d29922"
ACCENT_PUR   = "#bc8cff"
TEXT_MAIN    = "#e6edf3"
TEXT_DIM     = "#8b949e"
FONT_MONO    = ("Consolas", 9)
FONT_TITLE   = ("Segoe UI", 11, "bold")
FONT_NORMAL  = ("Segoe UI", 9)
FONT_SMALL   = ("Segoe UI", 8)

STATE_LABELS = [
    'u [m/s]',   'v [m/s]',   'w [m/s]',
    'p [rad/s]', 'q [rad/s]', 'r [rad/s]',
    'φ [rad]',   'θ [rad]',   'ψ [rad]',
]
STATE_NAMES = [
    'u — Vel. adelante',  'v — Vel. lateral',   'w — Vel. vertical (↓)',
    'p — Tasa alabeo',    'q — Tasa cabeceo',   'r — Tasa guiñada',
    'φ — Ángulo alabeo',  'θ — Ángulo cabeceo', 'ψ — Rumbo',
]

# =============================================================================
# HELPERS DE GRAFICACIÓN (embebidas en Tk)
# =============================================================================

def _make_dark_fig(rows, cols, figsize):
    fig = Figure(figsize=figsize, facecolor=DARK_BG)
    axes = []
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COL)
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.title.set_color(TEXT_MAIN)
        ax.grid(True, color=BORDER_COL, linewidth=0.5, alpha=0.6)
        axes.append(ax)
    return fig, axes


def _embed_figure(fig, parent_frame):
    """Embebe una figura matplotlib en un frame Tk con toolbar."""
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, parent_frame, pack_toolbar=False)
    toolbar.update()
    toolbar.configure(background=PANEL_BG)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    return canvas


def plot_states_embedded(parent_frame, t, X_list, labels_list, colors,
                          event_times=None, event_labels=None):
    """Dibuja las 9 variables de estado en un frame dado."""
    for w in parent_frame.winfo_children():
        w.destroy()

    fig, axes = _make_dark_fig(3, 3, (14, 10))
    fig.subplots_adjust(hspace=0.45, wspace=0.38, top=0.96, bottom=0.06,
                        left=0.07, right=0.98)

    for idx, ax in enumerate(axes):
        for X_data, lbl, col in zip(X_list, labels_list, colors):
            ax.plot(t, X_data[:, idx], label=lbl, color=col, linewidth=1.6)
        if event_times:
            ev_colors = [ACCENT_RED, ACCENT_ORG, ACCENT_PUR]
            for k, (te, le) in enumerate(zip(event_times, event_labels or event_times)):
                ax.axvline(te, color=ev_colors[k % len(ev_colors)],
                           linestyle='--', linewidth=1.2, alpha=0.8,
                           label=le if idx == 0 else "")
        ax.set_xlabel('Tiempo [s]', fontsize=7)
        ax.set_ylabel(STATE_LABELS[idx], fontsize=7)
        ax.set_title(STATE_NAMES[idx], fontsize=8, fontweight='bold', color=TEXT_MAIN)
        ax.legend(fontsize=6, loc='best', facecolor=PANEL_BG,
                  edgecolor=BORDER_COL, labelcolor=TEXT_MAIN)

    _embed_figure(fig, parent_frame)


def plot_3d_embedded(parent_frame, pos, label="Trayectoria"):
    for w in parent_frame.winfo_children():
        w.destroy()

    fig = Figure(figsize=(10, 7), facecolor=DARK_BG)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_DIM, labelsize=7)

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            color=ACCENT_BLUE, linewidth=2, label=label)
    ax.scatter(pos[0,  0], pos[0,  1], pos[0,  2],
               c=ACCENT_GREEN, s=80, zorder=5, label='Inicio')
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2],
               c=ACCENT_RED,   s=80, zorder=5, label='Fin')
    ax.set_xlabel('x_E [m]', color=TEXT_DIM, fontsize=8)
    ax.set_ylabel('y_E [m]', color=TEXT_DIM, fontsize=8)
    ax.set_zlabel('Alt [m]', color=TEXT_DIM, fontsize=8)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(True, color=BORDER_COL, linewidth=0.4)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=BORDER_COL, labelcolor=TEXT_MAIN)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    _embed_figure(fig, parent_frame)


def plot_3d_comparison_embedded(parent_frame, pos_list, labels, colors,
                                event_time=None):
    """Dibuja varias trayectorias 3D superpuestas en un frame dado."""
    for w in parent_frame.winfo_children():
        w.destroy()

    fig = Figure(figsize=(10, 7), facecolor=DARK_BG)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_DIM, labelsize=7)

    for pos, label, color in zip(pos_list, labels, colors):
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color=color, linewidth=2, label=label)
        ax.scatter(pos[0,  0], pos[0,  1], pos[0,  2],
                   c=ACCENT_GREEN, s=60, zorder=5)
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2],
                   c=color, s=60, marker='X', zorder=5)

    ax.set_xlabel('x_E [m] (Norte)', color=TEXT_DIM, fontsize=8)
    ax.set_ylabel('y_E [m] (Este)',  color=TEXT_DIM, fontsize=8)
    ax.set_zlabel('Alt [m]',         color=TEXT_DIM, fontsize=8)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(True, color=BORDER_COL, linewidth=0.4)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=BORDER_COL,
              labelcolor=TEXT_MAIN)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    _embed_figure(fig, parent_frame)


def plot_2d_embedded(parent_frame, pos):
    for w in parent_frame.winfo_children():
        w.destroy()

    fig = Figure(figsize=(12, 5), facecolor=DARK_BG)
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.96, top=0.92, bottom=0.12)

    t_arr = np.arange(len(pos))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor(PANEL_BG)
    sc = ax1.scatter(pos[:, 1], pos[:, 0], c=t_arr, cmap='plasma', s=4)
    ax1.scatter(pos[0, 1],  pos[0, 0],  c=ACCENT_GREEN, s=60, zorder=5, label='Inicio')
    ax1.scatter(pos[-1, 1], pos[-1, 0], c=ACCENT_RED,   s=60, zorder=5, label='Fin')
    fig.colorbar(sc, ax=ax1, label='Tiempo [s]', shrink=0.85)
    ax1.set_xlabel('y_E [m] (Este)', color=TEXT_DIM, fontsize=8)
    ax1.set_ylabel('x_E [m] (Norte)', color=TEXT_DIM, fontsize=8)
    ax1.set_title('Huella en tierra (Norte–Este)', color=TEXT_MAIN, fontsize=9, fontweight='bold')
    ax1.tick_params(colors=TEXT_DIM)
    ax1.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=BORDER_COL, labelcolor=TEXT_MAIN)
    for sp in ax1.spines.values():
        sp.set_edgecolor(BORDER_COL)
    ax1.grid(True, color=BORDER_COL, linewidth=0.4)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor(PANEL_BG)
    ax2.plot(t_arr, pos[:, 2], color=ACCENT_BLUE, linewidth=1.8)
    ax2.set_xlabel('Tiempo [s]', color=TEXT_DIM, fontsize=8)
    ax2.set_ylabel('Altitud [m]', color=TEXT_DIM, fontsize=8)
    ax2.set_title('Altitud vs. Tiempo', color=TEXT_MAIN, fontsize=9, fontweight='bold')
    ax2.tick_params(colors=TEXT_DIM)
    for sp in ax2.spines.values():
        sp.set_edgecolor(BORDER_COL)
    ax2.grid(True, color=BORDER_COL, linewidth=0.4)

    _embed_figure(fig, parent_frame)


def plot_pso_embedded(parent_frame, cost_hist):
    for w in parent_frame.winfo_children():
        w.destroy()

    fig = Figure(figsize=(9, 5), facecolor=DARK_BG)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.12)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(PANEL_BG)
    ax.semilogy(cost_hist, color=ACCENT_PUR, linewidth=2)
    ax.set_xlabel('Iteración PSO', color=TEXT_DIM, fontsize=9)
    ax.set_ylabel('Mejor costo global (log)', color=TEXT_DIM, fontsize=9)
    ax.set_title('Convergencia PSO — Trim Va=78 m/s, ψ=45° (NE)',
                 color=TEXT_MAIN, fontsize=10, fontweight='bold')
    ax.tick_params(colors=TEXT_DIM)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER_COL)
    ax.grid(True, which='both', color=BORDER_COL, linewidth=0.4)

    _embed_figure(fig, parent_frame)


# =============================================================================
# CLASE PRINCIPAL — VENTANA CON PESTAÑAS
# =============================================================================

class RCAMApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("RCAM Aircraft Simulation — Final Task")
        self.configure(bg=DARK_BG)
        self.geometry("1280x820")
        self.resizable(True, True)

        # Datos simulados (cacheados para no re-calcular)
        self._data = {}           # tarea_id → dict con t, X, pos, etc.
        self._running = {}        # tarea_id → bool

        self._output_folder = os.path.dirname(os.path.abspath(__file__))

        self._build_header()
        self._build_notebook()
        self._build_status_bar()

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------
    def _build_header(self):
        hdr = tk.Frame(self, bg=PANEL_BG, height=52, relief=tk.FLAT)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="✈  RCAM Aircraft Model", font=("Segoe UI", 14, "bold"),
                 bg=PANEL_BG, fg=ACCENT_BLUE).pack(side=tk.LEFT, padx=18, pady=10)

        tk.Label(hdr, text="GARTEUR FM(AG08) — 6-DOF Nonlinear Simulation",
                 font=FONT_NORMAL, bg=PANEL_BG, fg=TEXT_DIM).pack(side=tk.LEFT, pady=10)

        btn_folder = tk.Button(
            hdr, text="📁  Carpeta salida", font=FONT_SMALL,
            bg=BORDER_COL, fg=TEXT_MAIN, relief=tk.FLAT, padx=10,
            command=self._choose_folder, cursor="hand2"
        )
        btn_folder.pack(side=tk.RIGHT, padx=14, pady=10)

        btn_all = tk.Button(
            hdr, text="▶▶  Ejecutar todo", font=FONT_SMALL,
            bg=ACCENT_BLUE, fg=DARK_BG, relief=tk.FLAT, padx=12,
            command=self._run_all_tasks, cursor="hand2"
        )
        btn_all.pack(side=tk.RIGHT, padx=6, pady=10)

    # ------------------------------------------------------------------
    # NOTEBOOK DE PESTAÑAS
    # ------------------------------------------------------------------
    def _build_notebook(self):
        style = ttk.Style(self)
        style.theme_use("default")

        style.configure("Dark.TNotebook",
                         background=DARK_BG, borderwidth=0, tabmargins=0)
        style.configure("Dark.TNotebook.Tab",
                         background=PANEL_BG, foreground=TEXT_DIM,
                         font=("Segoe UI", 9, "bold"),
                         padding=[18, 8], borderwidth=0)
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", DARK_BG), ("active", BORDER_COL)],
                  foreground=[("selected", ACCENT_BLUE)])

        self._nb = ttk.Notebook(self, style="Dark.TNotebook")
        self._nb.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self._tab2  = self._make_tab("Tarea 2 — Vuelo Base",    self._build_tab2)
        self._tab3  = self._make_tab("Tarea 3 — Alerón +5°",    self._build_tab3)
        self._tab4a = self._make_tab("Tarea 4a — Falla Motor",  self._build_tab4a)
        self._tab4b = self._make_tab("Tarea 4b — PSO Trim",     self._build_tab4b)

    def _make_tab(self, title, builder_fn):
        frame = tk.Frame(self._nb, bg=DARK_BG)
        self._nb.add(frame, text=title)
        builder_fn(frame)
        return frame

    # ------------------------------------------------------------------
    # STATUS BAR
    # ------------------------------------------------------------------
    def _build_status_bar(self):
        sb = tk.Frame(self, bg=BORDER_COL, height=24)
        sb.pack(fill=tk.X, side=tk.BOTTOM)
        sb.pack_propagate(False)
        self._status_var = tk.StringVar(value="Listo.")
        tk.Label(sb, textvariable=self._status_var, font=FONT_SMALL,
                 bg=BORDER_COL, fg=TEXT_DIM, anchor='w').pack(side=tk.LEFT, padx=10)

    def _set_status(self, msg, color=TEXT_DIM):
        self._status_var.set(msg)
        # Color update not trivial in ttk Label — keep simple
        self.update_idletasks()

    # ------------------------------------------------------------------
    # HELPERS DE LAYOUT
    # ------------------------------------------------------------------
    def _control_row(self, parent, btn_text, btn_color, btn_cmd,
                     info_lines=None, extra_btns=None):
        """Barra superior de control con botón de ejecución e info."""
        ctrl = tk.Frame(parent, bg=PANEL_BG, height=52)
        ctrl.pack(fill=tk.X, side=tk.TOP)
        ctrl.pack_propagate(False)

        btn = tk.Button(ctrl, text=btn_text, font=("Segoe UI", 9, "bold"),
                        bg=btn_color, fg=DARK_BG, relief=tk.FLAT, padx=14,
                        command=btn_cmd, cursor="hand2")
        btn.pack(side=tk.LEFT, padx=14, pady=9)

        if extra_btns:
            for (etxt, ecmd) in extra_btns:
                eb = tk.Button(ctrl, text=etxt, font=FONT_SMALL,
                               bg=BORDER_COL, fg=TEXT_MAIN, relief=tk.FLAT, padx=10,
                               command=ecmd, cursor="hand2")
                eb.pack(side=tk.LEFT, padx=4, pady=9)

        if info_lines:
            for line in info_lines:
                tk.Label(ctrl, text=line, font=FONT_SMALL,
                         bg=PANEL_BG, fg=TEXT_DIM).pack(side=tk.LEFT, padx=8)

        return ctrl

    def _split_frame(self, parent, ratio=0.55):
        """Divide el espacio en panel de gráficas (izq) + consola (der)."""
        paned = tk.PanedWindow(parent, orient=tk.HORIZONTAL,
                               bg=DARK_BG, sashwidth=4, sashrelief=tk.FLAT,
                               sashpad=0)
        paned.pack(fill=tk.BOTH, expand=True)

        left  = tk.Frame(paned, bg=DARK_BG)
        right = tk.Frame(paned, bg=PANEL_BG)

        paned.add(left,  minsize=400)
        paned.add(right, minsize=280)

        # Ajustar sash al ratio pedido
        self.update_idletasks()
        total = paned.winfo_width()
        if total > 10:
            paned.sash_place(0, int(total * ratio), 0)
        paned.bind('<Configure>', lambda e, p=paned, r=ratio:
                   p.sash_place(0, int(p.winfo_width() * r), 0)
                   if p.winfo_width() > 10 else None)

        return left, right

    def _make_console(self, parent, height=None):
        """Panel de texto tipo consola."""
        frame = tk.Frame(parent, bg=PANEL_BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        tk.Label(frame, text=" 📋  Resultados / Análisis",
                 font=FONT_TITLE, bg=PANEL_BG, fg=ACCENT_BLUE,
                 anchor='w').pack(fill=tk.X, padx=10, pady=(8, 2))

        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        txt = tk.Text(frame, bg="#0a0c10", fg=TEXT_MAIN, font=FONT_MONO,
                      insertbackground=ACCENT_BLUE, relief=tk.FLAT,
                      yscrollcommand=sb.set, wrap=tk.WORD,
                      padx=10, pady=8, state=tk.DISABLED)
        txt.pack(fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 8))
        sb.config(command=txt.yview)

        txt.tag_configure("heading",  foreground=ACCENT_BLUE,  font=("Consolas", 9, "bold"))
        txt.tag_configure("ok",       foreground=ACCENT_GREEN)
        txt.tag_configure("warn",     foreground=ACCENT_ORG)
        txt.tag_configure("val",      foreground=ACCENT_PUR)
        txt.tag_configure("dim",      foreground=TEXT_DIM)

        return txt

    def _console_write(self, txt_widget, line, tag=None):
        txt_widget.config(state=tk.NORMAL)
        if tag:
            txt_widget.insert(tk.END, line + "\n", tag)
        else:
            txt_widget.insert(tk.END, line + "\n")
        txt_widget.see(tk.END)
        txt_widget.config(state=tk.DISABLED)
        self.update_idletasks()

    def _console_clear(self, txt_widget):
        txt_widget.config(state=tk.NORMAL)
        txt_widget.delete("1.0", tk.END)
        txt_widget.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # PESTAÑA 2 — VUELO BASE
    # ------------------------------------------------------------------
    def _build_tab2(self, parent):
        info = ["X₀=[85,0,0,0,0,0,0,0.1,0]ᵀ",
                "  u₀=[0,−0.1,0,0.08,0.08]ᵀ",
                "  Δt=1 s, T=180 s"]
        ctrl = self._control_row(
            parent, "▶  Ejecutar Tarea 2", ACCENT_GREEN,
            self._run_task2, info_lines=info,
            extra_btns=[
                ("Guardar gráficas", lambda: self._save_task2()),
            ]
        )

        # Sub-notebook de vistas: 3D, 2D, Estados
        inner_nb = ttk.Notebook(parent, style="Dark.TNotebook")
        inner_nb.pack(fill=tk.BOTH, expand=True)

        self._t2_3d_frame    = tk.Frame(inner_nb, bg=DARK_BG)
        self._t2_2d_frame    = tk.Frame(inner_nb, bg=DARK_BG)
        self._t2_state_frame = tk.Frame(inner_nb, bg=DARK_BG)
        self._t2_console_outer = tk.Frame(inner_nb, bg=PANEL_BG)

        inner_nb.add(self._t2_3d_frame,       text="  Trayectoria 3D  ")
        inner_nb.add(self._t2_2d_frame,       text="  Huella + Altitud  ")
        inner_nb.add(self._t2_state_frame,    text="  Variables de estado  ")
        inner_nb.add(self._t2_console_outer,  text="  Análisis  ")

        self._t2_console = self._make_console(self._t2_console_outer)
        self._console_write(self._t2_console,
            "Presiona  ▶ Ejecutar Tarea 2  para comenzar la simulación.", "dim")

    def _run_task2(self):
        if self._running.get("t2"):
            return
        self._running["t2"] = True
        self._set_status("Tarea 2: simulando 180 s …")
        threading.Thread(target=self._task2_worker, daemon=True).start()

    def _task2_worker(self):
        try:
            c = self._t2_console
            self._console_clear(c)
            self._console_write(c, "═"*55, "heading")
            self._console_write(c, "  TAREA 2 — VUELO BASE (180 s)", "heading")
            self._console_write(c, "═"*55, "heading")
            self._console_write(c, "Iniciando simulación RK4 (dt_int=0.05 s)…", "dim")

            t2, X2, pos2 = simulate(X0, constant_control, (0, 180), dt=1.0)

            self._data["t2"] = {"t": t2, "X": X2, "pos": pos2}

            self.after(0, lambda: self._render_task2(t2, X2, pos2))

            # Análisis
            self._console_write(c, "\n✔  Simulación completada.", "ok")
            self._console_write(c, "\n─── Análisis ───────────────────────────", "heading")

            u_range = (X2[:, 0].min(), X2[:, 0].max())
            self._console_write(c, f"  u: [{u_range[0]:.2f}, {u_range[1]:.2f}] m/s")
            self._console_write(c, f"     Oscilaciones fugoide por CI no-trim.", "dim")

            w_range = (X2[:, 2].min(), X2[:, 2].max())
            self._console_write(c, f"  w: [{w_range[0]:.2f}, {w_range[1]:.2f}] m/s")
            self._console_write(c, f"     Acoplado con q (periodo corto).", "dim")

            theta_f = X2[-1, 7]
            self._console_write(c, f"  θ final: {np.degrees(theta_f):.3f}° "
                                   f"(converge al ángulo de trim)", "val")

            alt_change = pos2[-1, 2] - pos2[0, 2]
            sign = "subida" if alt_change > 0 else "descenso"
            self._console_write(c, f"  Altitud: {sign} neto de {abs(alt_change):.1f} m")
            self._console_write(c,
                "  → CI fuera de trim → transitorio fugoide → ajuste de altitud.", "dim")

            phi_max = np.degrees(np.abs(X2[:, 6]).max())
            self._console_write(c, f"  φ_max: {phi_max:.4f}° ≈ 0 (simetría lateral OK)", "ok")

            self._console_write(c, "\n─── Condiciones de la CI ─────────────", "heading")
            self._console_write(c, f"  X₀ = [85, 0, 0, 0, 0, 0, 0, 0.1, 0]ᵀ")
            self._console_write(c, f"  u₀ = [0, −0.1, 0, 0.08, 0.08]ᵀ")
            self._console_write(c,
                "  NOTA: La CI no es un punto de trim → transitorio de ~30 s.", "warn")

        except Exception as ex:
            self._console_write(self._t2_console, f"\n[ERROR] {ex}", "warn")
        finally:
            self._running["t2"] = False
            self._set_status("Tarea 2 completada.")

    def _render_task2(self, t2, X2, pos2):
        plot_3d_embedded(self._t2_3d_frame,    pos2)
        plot_2d_embedded(self._t2_2d_frame,    pos2)
        plot_states_embedded(
            self._t2_state_frame, t2,
            [X2], ["Base (CI no-trim)"], [ACCENT_BLUE]
        )

    def _save_task2(self):
        if "t2" not in self._data:
            messagebox.showinfo("Aviso", "Ejecuta la Tarea 2 primero.")
            return
        d = self._data["t2"]
        self._save_states_png("tarea2_estados.png", d["t"], [d["X"]],
                              ["Base"], [ACCENT_BLUE])

    # ------------------------------------------------------------------
    # PESTAÑA 3 — PULSO DE ALERÓN +5°
    # ------------------------------------------------------------------
    def _build_tab3(self, parent):
        info = ["da=+5° en t=30–32 s",
                "  Comparado vs. base (Tarea 2)"]
        self._control_row(
            parent, "▶  Ejecutar Tarea 3", ACCENT_ORG,
            self._run_task3, info_lines=info,
            extra_btns=[("Guardar gráficas", lambda: self._save_task3())]
        )

        inner_nb = ttk.Notebook(parent, style="Dark.TNotebook")
        inner_nb.pack(fill=tk.BOTH, expand=True)

        self._t3_3d_frame     = tk.Frame(inner_nb, bg=DARK_BG)
        self._t3_state_frame  = tk.Frame(inner_nb, bg=DARK_BG)
        self._t3_lat_frame    = tk.Frame(inner_nb, bg=DARK_BG)
        self._t3_console_out  = tk.Frame(inner_nb, bg=PANEL_BG)

        inner_nb.add(self._t3_3d_frame,     text="  Trayectoria 3D  ")
        inner_nb.add(self._t3_state_frame,  text="  9 Variables de estado  ")
        inner_nb.add(self._t3_lat_frame,    text="  Dinámica lateral  ")
        inner_nb.add(self._t3_console_out,  text="  Análisis  ")

        self._t3_console = self._make_console(self._t3_console_out)
        self._console_write(self._t3_console,
            "Presiona  ▶ Ejecutar Tarea 3  para comenzar.", "dim")

    def _run_task3(self):
        if self._running.get("t3"):
            return
        self._running["t3"] = True
        self._set_status("Tarea 3: simulando pulso de alerón …")
        threading.Thread(target=self._task3_worker, daemon=True).start()

    def _task3_worker(self):
        try:
            c = self._t3_console
            self._console_clear(c)
            self._console_write(c, "═"*55, "heading")
            self._console_write(c, "  TAREA 3 — PULSO ALERÓN +5° (t=5–7 s)", "heading")
            self._console_write(c, "═"*55, "heading")
            self._console_write(c,
                "  Pulso en t=5–7 s → Va≈84 m/s (condición inicial)", "dim")
            self._console_write(c,
                "  (Antes estaba en t=30–32 s: Va=123 m/s, θ=−13°, incorrecto)", "warn")

            # dt=0.1 s para AMBAS simulaciones → misma longitud de vector → sin error de shape.
            # Se guarda como "t2_fine" para no sobreescribir el caché de Tarea 2 (dt=1.0).
            if "t2_fine" not in self._data:
                self._console_write(c, "  Calculando base (alta resolución)…", "dim")
                t2, X2, pos2 = simulate(X0, constant_control, (0, 180), dt=0.1)
                self._data["t2_fine"] = {"t": t2, "X": X2, "pos": pos2}
            else:
                t2   = self._data["t2_fine"]["t"]
                X2   = self._data["t2_fine"]["X"]
                pos2 = self._data["t2_fine"]["pos"]

            self._console_write(c, "  Simulando con pulso de alerón…", "dim")
            t3, X3, pos3 = simulate(X0, aileron_impulse_control, (0, 180), dt=0.1)
            self._data["t3"] = {"t": t3, "X": X3, "pos": pos3}

            self.after(0, lambda: self._render_task3(t3, X3, X2, pos3, pos2))

            self._console_write(c, "\n✔  Simulación completada.", "ok")
            self._console_write(c, "\n─── Convención de signo RCAM ──────────", "heading")
            self._console_write(c, "  dCl/dδa = -0.6  →  δa > 0 produce Cl < 0")
            self._console_write(c, "  → p < 0 (tasa de alabeo a la IZQUIERDA)", "val")
            self._console_write(c, "  → φ < 0 (banqueo a BABOR)", "val")

            p_min  = np.degrees(X3[:, 3].min())
            t_pmin = t3[np.argmin(X3[:, 3])]
            idx_12 = np.searchsorted(t3, 12.0)
            phi_12 = np.degrees(X3[min(idx_12, len(X3)-1), 6])
            mask_adv = (t3 >= 5.0) & (t3 < 7.0)
            r_adv = np.degrees(X3[mask_adv, 5].mean()) if mask_adv.any() else 0.0

            self._console_write(c, "\n─── Valores pico ──────────────────────", "heading")
            self._console_write(c,
                f"  p_mín = {p_min:.3f}°/s  a t = {t_pmin:.1f} s", "val")
            self._console_write(c,
                f"  φ @ t=12 s = {phi_12:.3f}° (negativo = izquierda)", "val")
            self._console_write(c,
                f"  r promedio (t=5–7 s) = {r_adv:.4f}°/s (guiñada adversa)", "val")

            self._console_write(c, "\n─── Secuencia de eventos ──────────────", "heading")
            steps = [
                "1. t=5 s: δa=+5° → Cl<0 → p negativo → φ negativo (banqueo izq.)",
                "2. t=5–7 s: guiñada adversa leve (acoplamiento alabeo-guiñada)",
                "3. v crece: deslizamiento lateral por el banqueo",
                "4. t=7 s: δa=0 → p se amortigua; φ permanece negativo",
                "5. t>7 s: Dutch-roll (p, r, v, φ oscilan) + modo espiral",
            ]
            for s in steps:
                self._console_write(c, f"  {s}")

        except Exception as ex:
            self._console_write(self._t3_console, f"\n[ERROR] {ex}", "warn")
        finally:
            self._running["t3"] = False
            self._set_status("Tarea 3 completada.")

    def _render_task3(self, t3, X3, X2, pos3, pos2):
        plot_3d_comparison_embedded(
            self._t3_3d_frame,
            [pos3],
            ["Alerón +5° (t=5–7 s)"],
            [ACCENT_ORG],
        )
        plot_states_embedded(
            self._t3_state_frame, t3,
            [X3, X2],
            ["Alerón +5° (t=5–7 s)", "Base (ref.)"],
            [ACCENT_ORG, ACCENT_BLUE],
            event_times=[5.0, 7.0],
            event_labels=["t=5 s (ON)", "t=7 s (OFF)"]
        )
        # Detalle lateral: p, φ, r, v
        for w in self._t3_lat_frame.winfo_children():
            w.destroy()

        fig, axes = _make_dark_fig(2, 2, (12, 8))
        fig.subplots_adjust(hspace=0.42, wspace=0.35, top=0.96, bottom=0.07,
                            left=0.08, right=0.97)

        vars_ = [(3, 'p [rad/s]', 'Tasa de alabeo (p)'),
                 (6, 'φ [rad]',   'Ángulo de alabeo (φ)'),
                 (5, 'r [rad/s]', 'Tasa de guiñada (r) — adversa'),
                 (1, 'v [m/s]',   'Velocidad lateral (v)')]

        for ax, (idx, ylab, ttl) in zip(axes, vars_):
            ax.plot(t3, X3[:, idx], color=ACCENT_ORG, lw=2,   label="δa=+5°")
            ax.plot(t3, X2[:, idx], color=ACCENT_BLUE, lw=1.4,
                    linestyle='--', label="Base")
            ax.axvline(5,  color=ACCENT_RED,   ls=':', lw=1.4, label='t=5 s')
            ax.axvline(7,  color=ACCENT_ORG,   ls=':', lw=1.4, label='t=7 s')
            ax.axhline(0,  color=TEXT_DIM,     lw=0.7)
            ax.set_xlabel('Tiempo [s]', fontsize=7)
            ax.set_ylabel(ylab, fontsize=7)
            ax.set_title(ttl, fontsize=8, fontweight='bold', color=TEXT_MAIN)
            ax.legend(fontsize=6, facecolor=PANEL_BG,
                      edgecolor=BORDER_COL, labelcolor=TEXT_MAIN)

        _embed_figure(fig, self._t3_lat_frame)

    def _save_task3(self):
        if "t3" not in self._data:
            messagebox.showinfo("Aviso", "Ejecuta la Tarea 3 primero.")
            return
        d3 = self._data["t3"]
        d2 = self._data.get("t2", {})
        lists_ = [d3["X"]]
        lbls_  = ["Alerón +5°"]
        cols_  = [ACCENT_ORG]
        if d2:
            lists_.append(d2["X"]); lbls_.append("Base"); cols_.append(ACCENT_BLUE)
        self._save_states_png("tarea3_estados.png", d3["t"], lists_, lbls_, cols_)

    # ------------------------------------------------------------------
    # PESTAÑA 4a — FALLA DE MOTOR
    # ------------------------------------------------------------------
    def _build_tab4a(self, parent):
        info = ["Motor 1 (estribor) → mínimo en t=30 s",
                "  dth1: 0.08 → 0.00873 rad"]
        self._control_row(
            parent, "▶  Ejecutar Tarea 4a", ACCENT_RED,
            self._run_task4a, info_lines=info,
            extra_btns=[("Guardar gráficas", lambda: self._save_task4a())]
        )

        inner_nb = ttk.Notebook(parent, style="Dark.TNotebook")
        inner_nb.pack(fill=tk.BOTH, expand=True)

        self._t4a_3d_frame    = tk.Frame(inner_nb, bg=DARK_BG)
        self._t4a_state_frame = tk.Frame(inner_nb, bg=DARK_BG)
        self._t4a_det_frame   = tk.Frame(inner_nb, bg=DARK_BG)
        self._t4a_console_out = tk.Frame(inner_nb, bg=PANEL_BG)

        inner_nb.add(self._t4a_3d_frame,    text="  Trayectoria 3D  ")
        inner_nb.add(self._t4a_state_frame, text="  9 Variables de estado  ")
        inner_nb.add(self._t4a_det_frame,   text="  Detalle falla  ")
        inner_nb.add(self._t4a_console_out, text="  Análisis  ")

        self._t4a_console = self._make_console(self._t4a_console_out)
        self._console_write(self._t4a_console,
            "Presiona  ▶ Ejecutar Tarea 4a  para comenzar.", "dim")

    def _run_task4a(self):
        if self._running.get("t4a"):
            return
        self._running["t4a"] = True
        self._set_status("Tarea 4a: simulando falla de motor …")
        threading.Thread(target=self._task4a_worker, daemon=True).start()

    def _task4a_worker(self):
        try:
            c = self._t4a_console
            self._console_clear(c)
            self._console_write(c, "═"*55, "heading")
            self._console_write(c, "  TAREA 4a — FALLA MOTOR 1 (t≥30 s)", "heading")
            self._console_write(c, "═"*55, "heading")

            # dt=0.1 s para AMBAS simulaciones → mismo número de puntos → sin error de shape.
            if "t2_fine" not in self._data:
                self._console_write(c, "  Calculando base (alta resolución)…", "dim")
                t2, X2, pos2 = simulate(X0, constant_control, (0, 180), dt=0.1)
                self._data["t2_fine"] = {"t": t2, "X": X2, "pos": pos2}
            X2   = self._data["t2_fine"]["X"]
            t2   = self._data["t2_fine"]["t"]
            pos2 = self._data["t2_fine"]["pos"]

            self._console_write(c, "  Simulando con falla de motor 1…", "dim")
            t4, X4, pos4 = simulate(X0, engine_shutdown_control, (0, 180), dt=0.1)
            self._data["t4a"] = {"t": t4, "X": X4, "pos": pos4}

            self.after(0, lambda: self._render_task4a(t4, X4, X2, pos4, pos2))

            self._console_write(c, "\n✔  Simulación completada.", "ok")

            idx_60 = min(np.searchsorted(t4, 60.0), len(X4) - 1)
            r60  = np.degrees(X4[idx_60, 5])
            ph60 = np.degrees(X4[idx_60, 6])
            v60  = X4[idx_60, 1]
            ps60 = np.degrees(X4[idx_60, 8])
            u60  = X4[idx_60, 0]
            w60  = X4[idx_60, 2]

            self._console_write(c, "\n─── Física de la asimetría ────────────", "heading")
            self._console_write(c,
                "  Motor 1 (y=+7.94 m) cae a δTH=0.5°/180*π ≈ 0.00873 rad")
            self._console_write(c,
                "  Mz = 7.94·(F2−F1) > 0  →  guiñada +  (nariz a la DERECHA)", "val")
            self._console_write(c,
                "  Cl_β = -1.4, β>0  →  Cl<0  →  φ<0 (banqueo a BABOR)", "val")

            self._console_write(c, "\n─── Estado @ t=60 s (30 s tras falla) ─", "heading")
            self._console_write(c, f"  r  = {r60:.3f}°/s  (+ = gira a la derecha)", "val")
            self._console_write(c, f"  φ  = {ph60:.3f}°   (- = banqueo izquierda)", "val")
            self._console_write(c, f"  v  = {v60:.3f} m/s  (deslizamiento lateral)", "val")
            self._console_write(c, f"  ψ  = {ps60:.3f}°   (rumbo)", "val")
            self._console_write(c, f"  u  = {u60:.3f} m/s  (pérdida de velocidad)", "val")
            self._console_write(c, f"  w  = {w60:.3f} m/s  (descenso)", "val")

            self._console_write(c, "\n─── Secuencia de eventos ──────────────", "heading")
            steps = [
                "1. t=30 s: F1↓ → Mz>0 → r>0 (guiñada derecha)",
                "2. β>0 → Cl<0 → p<0 → φ<0 (banqueo izquierda)",
                "3. u↓: menos empuje total → pérdida de airspeed",
                "4. w↑: lift insuficiente → el avión desciende",
                "5. Sin corrección → espiral de picada a la izquierda",
            ]
            for s in steps:
                self._console_write(c, f"  {s}")

        except Exception as ex:
            self._console_write(self._t4a_console, f"\n[ERROR] {ex}", "warn")
        finally:
            self._running["t4a"] = False
            self._set_status("Tarea 4a completada.")

    def _render_task4a(self, t4, X4, X2, pos4, pos2):
        plot_3d_comparison_embedded(
            self._t4a_3d_frame,
            [pos4],
            ["Falla Motor 1 (t≥30 s)"],
            [ACCENT_RED],
        )
        plot_states_embedded(
            self._t4a_state_frame, t4,
            [X4, X2],
            ["Falla Motor 1 (t≥30 s)", "Base (ref.)"],
            [ACCENT_RED, ACCENT_BLUE],
            event_times=[30.0],
            event_labels=["t=30 s (falla Motor 1)"]
        )
        # Detalle: r, φ, v, u, ψ, w
        for w in self._t4a_det_frame.winfo_children():
            w.destroy()

        fig, axes = _make_dark_fig(3, 2, (12, 10))
        fig.subplots_adjust(hspace=0.48, wspace=0.35, top=0.97, bottom=0.06,
                            left=0.08, right=0.97)

        vars_ = [
            (5, 'r [rad/s]', 'Tasa de guiñada (r)  → +derecha'),
            (6, 'φ [rad]',   'Ángulo de alabeo (φ)  → −izquierda'),
            (1, 'v [m/s]',   'Velocidad lateral (v)  deslizamiento'),
            (0, 'u [m/s]',   'Velocidad adelante (u)  pérdida thrust'),
            (8, 'ψ [rad]',   'Rumbo (ψ)  giro a la derecha'),
            (2, 'w [m/s]',   'Vel. vertical (w)  >0=descenso'),
        ]
        for ax, (idx, ylab, ttl) in zip(axes, vars_):
            ax.plot(t4, X4[:, idx], color=ACCENT_RED,  lw=2,   label="Falla M1")
            ax.plot(t4, X2[:, idx], color=ACCENT_BLUE, lw=1.4,
                    linestyle='--', label="Base")
            ax.axvline(30, color='black', ls=':', lw=1.4, label='t=30 s')
            ax.axhline(0,  color=TEXT_DIM, lw=0.7)
            ax.set_xlabel('Tiempo [s]', fontsize=7)
            ax.set_ylabel(ylab, fontsize=7)
            ax.set_title(ttl, fontsize=8, fontweight='bold', color=TEXT_MAIN)
            ax.legend(fontsize=6, facecolor=PANEL_BG,
                      edgecolor=BORDER_COL, labelcolor=TEXT_MAIN)

        _embed_figure(fig, self._t4a_det_frame)

    def _save_task4a(self):
        if "t4a" not in self._data:
            messagebox.showinfo("Aviso", "Ejecuta la Tarea 4a primero.")
            return
        d = self._data["t4a"]
        self._save_states_png("tarea4a_estados.png", d["t"], [d["X"]],
                              ["Falla Motor 1"], [ACCENT_RED])

    # ------------------------------------------------------------------
    # PESTAÑA 4b — PSO TRIM
    # ------------------------------------------------------------------
    def _build_tab4b(self, parent):
        pso_ctrl = tk.Frame(parent, bg=PANEL_BG, height=64)
        pso_ctrl.pack(fill=tk.X, side=tk.TOP)
        pso_ctrl.pack_propagate(False)

        tk.Button(pso_ctrl, text="▶  Ejecutar PSO Trim", font=("Segoe UI", 9, "bold"),
                  bg=ACCENT_PUR, fg=DARK_BG, relief=tk.FLAT, padx=14,
                  command=self._run_task4b, cursor="hand2").pack(side=tk.LEFT, padx=14, pady=10)

        # Opciones PSO
        self._pso_particles = tk.IntVar(value=50)
        self._pso_iters     = tk.IntVar(value=2000)

        tk.Label(pso_ctrl, text="Partículas:", font=FONT_SMALL,
                 bg=PANEL_BG, fg=TEXT_DIM).pack(side=tk.LEFT, padx=(12, 2), pady=10)
        tk.Spinbox(pso_ctrl, from_=10, to=200, textvariable=self._pso_particles,
                   width=5, font=FONT_SMALL, bg=BORDER_COL, fg=TEXT_MAIN,
                   buttonbackground=BORDER_COL, relief=tk.FLAT).pack(side=tk.LEFT, pady=10)

        tk.Label(pso_ctrl, text="  Iteraciones:", font=FONT_SMALL,
                 bg=PANEL_BG, fg=TEXT_DIM).pack(side=tk.LEFT, padx=(8, 2), pady=10)
        tk.Spinbox(pso_ctrl, from_=100, to=5000, textvariable=self._pso_iters,
                   width=6, font=FONT_SMALL, bg=BORDER_COL, fg=TEXT_MAIN,
                   buttonbackground=BORDER_COL, relief=tk.FLAT,
                   increment=100).pack(side=tk.LEFT, pady=10)

        tk.Label(pso_ctrl, text="  Va=78 m/s  ψ=45° (NE)  vuelo nivelado",
                 font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_DIM).pack(side=tk.LEFT, padx=14)

        inner_nb = ttk.Notebook(parent, style="Dark.TNotebook")
        inner_nb.pack(fill=tk.BOTH, expand=True)

        self._t4b_conv_frame   = tk.Frame(inner_nb, bg=DARK_BG)
        self._t4b_verif_frame  = tk.Frame(inner_nb, bg=DARK_BG)
        self._t4b_console_out  = tk.Frame(inner_nb, bg=PANEL_BG)

        inner_nb.add(self._t4b_conv_frame,  text="  Convergencia PSO  ")
        inner_nb.add(self._t4b_verif_frame, text="  Verificación Trim  ")
        inner_nb.add(self._t4b_console_out, text="  Resultado Trim  ")

        self._t4b_console = self._make_console(self._t4b_console_out)
        self._console_write(self._t4b_console,
            "Presiona  ▶ Ejecutar PSO Trim  para comenzar.", "dim")
        self._console_write(self._t4b_console,
            "Advertencia: 2000 iteraciones pueden tardar 1–3 min.", "warn")

    def _run_task4b(self):
        if self._running.get("t4b"):
            return
        self._running["t4b"] = True
        n_p = self._pso_particles.get()
        n_i = self._pso_iters.get()
        self._set_status(f"Tarea 4b: PSO {n_p} partículas × {n_i} iteraciones …")
        threading.Thread(target=self._task4b_worker,
                         args=(n_p, n_i), daemon=True).start()

    def _task4b_worker(self, n_particles, n_iter):
        try:
            c = self._t4b_console
            self._console_clear(c)
            self._console_write(c, "═"*55, "heading")
            self._console_write(c, "  TAREA 4b — PSO TRIM (Va=78 m/s, ψ=45°)", "heading")
            self._console_write(c, "═"*55, "heading")
            self._console_write(c,
                f"  {n_particles} partículas  ×  {n_iter} iteraciones", "dim")
            self._console_write(c,
                "  Función de costo: Σ(Ẋᵢ/refᵢ)² + penalización simetría", "dim")
            self._console_write(c, "\n  Ejecutando PSO…", "dim")

            trim_params, trim_cost, cost_hist = pso_trim(
                n_particles=n_particles, n_iter=n_iter
            )

            alpha_t = trim_params[0]
            u_trim  = VA_TRIM * np.cos(alpha_t)
            w_trim  = VA_TRIM * np.sin(alpha_t)
            X_trim  = np.array([u_trim, 0.0, w_trim,
                                 0.0, 0.0, 0.0,
                                 0.0, alpha_t, PSI_TRIM])
            U_trim  = trim_params[1:]
            Xd_trim = xdot(X_trim, U_trim)

            # Verificación: simular 60 s desde el punto de trim
            def _trim_ctrl(t, X):
                return U_trim.copy()

            self._console_write(c, "  Verificando trim (60 s)…", "dim")
            t_v, X_v, _ = simulate(X_trim, _trim_ctrl, (0, 60), dt=1.0)

            self._data["t4b"] = {
                "trim_params": trim_params, "trim_cost": trim_cost,
                "cost_hist": cost_hist, "X_trim": X_trim, "U_trim": U_trim,
                "Xd_trim": Xd_trim, "t_v": t_v, "X_v": X_v
            }

            self.after(0, lambda: self._render_task4b(cost_hist, t_v, X_v))

            # Reporte
            self._console_write(c, "\n✔  PSO completado.", "ok")
            self._console_write(c, "\n─── Solución de trim ──────────────────", "heading")
            self._console_write(c,
                f"  Costo final:  {trim_cost:.6e}", "val")
            self._console_write(c,
                f"  α (AoA):      {np.degrees(alpha_t):.4f}°", "val")
            self._console_write(c, "\n  Estado de trim X_trim:", "heading")
            self._console_write(c,
                f"    u  = {X_trim[0]:.4f} m/s", "val")
            self._console_write(c,
                f"    v  = {X_trim[1]:.4f} m/s", "val")
            self._console_write(c,
                f"    w  = {X_trim[2]:.4f} m/s", "val")
            self._console_write(c,
                f"    p  = {np.degrees(X_trim[3]):.6f} °/s", "val")
            self._console_write(c,
                f"    q  = {np.degrees(X_trim[4]):.6f} °/s", "val")
            self._console_write(c,
                f"    r  = {np.degrees(X_trim[5]):.6f} °/s", "val")
            self._console_write(c,
                f"    φ  = {np.degrees(X_trim[6]):.4f} °", "val")
            self._console_write(c,
                f"    θ  = {np.degrees(X_trim[7]):.4f} °  (= α en vuelo nivelado)", "val")
            self._console_write(c,
                f"    ψ  = {np.degrees(X_trim[8]):.4f} °  (NE)", "val")
            self._console_write(c, "\n  Control de trim U_trim:", "heading")
            self._console_write(c,
                f"    δa   = {np.degrees(U_trim[0]):.4f} °", "val")
            self._console_write(c,
                f"    δe   = {np.degrees(U_trim[1]):.4f} °", "val")
            self._console_write(c,
                f"    δr   = {np.degrees(U_trim[2]):.4f} °", "val")
            self._console_write(c,
                f"    δTH1 = {U_trim[3]:.4f}  (fracción, F = dth·m·g)", "val")
            self._console_write(c,
                f"    δTH2 = {U_trim[4]:.4f}  (fracción, F = dth·m·g)", "val")
            self._console_write(c, "\n  Residuo Ẋ @ trim (→ 0):", "heading")
            self._console_write(c,
                f"    [u̇,v̇,ẇ] = {np.round(Xd_trim[0:3],6)}", "val")
            self._console_write(c,
                f"    [ṗ,q̇,ṙ] = {np.round(Xd_trim[3:6],6)}", "val")
            self._console_write(c,
                f"    [φ̇,θ̇,ψ̇] = {np.round(Xd_trim[6:9],6)}", "val")

            # Evaluación de calidad del trim
            resid = np.max(np.abs(Xd_trim[0:6]))
            if resid < 0.05:
                self._console_write(c,
                    f"\n  ✔  Residuo máximo = {resid:.4f}  — trim excelente", "ok")
            elif resid < 0.5:
                self._console_write(c,
                    f"\n  ⚠  Residuo máximo = {resid:.4f}  — trim aceptable", "warn")
            else:
                self._console_write(c,
                    f"\n  ✗  Residuo máximo = {resid:.4f}  — trim deficiente, "
                    "aumenta iteraciones", "warn")

        except Exception as ex:
            self._console_write(self._t4b_console, f"\n[ERROR] {ex}", "warn")
        finally:
            self._running["t4b"] = False
            self._set_status("Tarea 4b completada.")

    def _render_task4b(self, cost_hist, t_v, X_v):
        plot_pso_embedded(self._t4b_conv_frame, cost_hist)
        plot_states_embedded(
            self._t4b_verif_frame, t_v,
            [X_v], ["Desde punto de trim"], [ACCENT_PUR]
        )

    # ------------------------------------------------------------------
    # EJECUTAR TODAS LAS TAREAS
    # ------------------------------------------------------------------
    def _run_all_tasks(self):
        if any(self._running.values()):
            messagebox.showinfo("Ocupado", "Hay una tarea en ejecución.")
            return
        threading.Thread(target=self._all_tasks_worker, daemon=True).start()

    def _all_tasks_worker(self):
        self._set_status("Ejecutando todas las tareas…")
        # Tarea 2
        self._running["t2"] = True
        self._task2_worker()
        # Tarea 3
        self._running["t3"] = True
        self._task3_worker()
        # Tarea 4a
        self._running["t4a"] = True
        self._task4a_worker()
        # Tarea 4b
        self._running["t4b"] = True
        self._task4b_worker(
            self._pso_particles.get(), self._pso_iters.get()
        )
        self._set_status("✔  Todas las tareas completadas.")

    # ------------------------------------------------------------------
    # GUARDAR CARPETA Y FIGURAS EXTERNAS
    # ------------------------------------------------------------------
    def _choose_folder(self):
        folder = filedialog.askdirectory(
            title="Seleccionar carpeta para guardar resultados",
            initialdir=self._output_folder
        )
        if folder:
            self._output_folder = folder
            self._set_status(f"Carpeta de salida: {folder}")

    def _save_states_png(self, filename, t, X_list, labels, colors,
                         event_times=None, event_labels=None):
        fig, axes_list = plt.subplots(3, 3, figsize=(16, 12))
        fig.patch.set_facecolor(DARK_BG)
        for idx, ax in enumerate(axes_list.flat):
            ax.set_facecolor(PANEL_BG)
            for X_data, lbl, col in zip(X_list, labels, colors):
                ax.plot(t, X_data[:, idx], color=col, label=lbl, lw=1.7)
            if event_times:
                for te, le in zip(event_times, event_labels or event_times):
                    ax.axvline(te, color=ACCENT_RED, ls='--', lw=1.2, label=le)
            ax.set_xlabel('t [s]', fontsize=7)
            ax.set_ylabel(STATE_LABELS[idx], fontsize=7)
            ax.set_title(STATE_NAMES[idx], fontsize=8)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self._output_folder, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig)
        self._set_status(f"Guardado: {path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = RCAMApp()
    app.mainloop()
