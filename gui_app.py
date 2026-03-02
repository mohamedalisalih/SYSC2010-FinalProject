import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from io_loader import load_csv_numeric, SignalData
from analysis import make_time_axis, compute_fft


class SignalAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SYSC2010 Signal Analyzer")
        self.geometry("1000x700")

        # Stored data
        self.data: SignalData | None = None
        self.loaded_path: str | None = None

        # ===== Controls Frame =====
        controls = ttk.LabelFrame(self, text="Controls")
        controls.pack(fill="x", padx=10, pady=10)

        self.btn_load = ttk.Button(controls, text="Load CSV", command=self.on_load_csv)
        self.btn_load.grid(row=0, column=0, padx=8, pady=8, sticky="w")

        ttk.Label(controls, text="Signal Type:").grid(row=0, column=1, padx=8, pady=8, sticky="e")
        self.signal_type = tk.StringVar(value="ECG")
        ttk.Combobox(
            controls,
            textvariable=self.signal_type,
            values=["ECG", "Respiration", "Temperature", "Motion"],
            state="readonly",
            width=15
        ).grid(row=0, column=2, padx=8, pady=8, sticky="w")

        ttk.Label(controls, text="Filter:").grid(row=0, column=3, padx=8, pady=8, sticky="e")
        self.filter_type = tk.StringVar(value="None")
        ttk.Combobox(
            controls,
            textvariable=self.filter_type,
            values=["None", "LPF", "HPF", "BPF"],
            state="readonly",
            width=8
        ).grid(row=0, column=4, padx=8, pady=8, sticky="w")

        ttk.Label(controls, text="Method:").grid(row=0, column=5, padx=8, pady=8, sticky="e")
        self.method = tk.StringVar(value="FIR")
        ttk.Combobox(
            controls,
            textvariable=self.method,
            values=["FIR", "IIR"],
            state="readonly",
            width=6
        ).grid(row=0, column=6, padx=8, pady=8, sticky="w")

        self.btn_apply = ttk.Button(controls, text="Apply", command=self.on_apply)
        self.btn_apply.grid(row=0, column=7, padx=8, pady=8, sticky="w")

        self.btn_reset = ttk.Button(controls, text="Reset", command=self.on_reset)
        self.btn_reset.grid(row=0, column=8, padx=8, pady=8, sticky="w")

        controls.columnconfigure(9, weight=1)

        # ===== Status Frame =====
        status = ttk.LabelFrame(self, text="Loaded Data Status")
        status.pack(fill="x", padx=10, pady=(0, 10))

        self.file_label_var = tk.StringVar(value="File: (none)")
        self.samples_label_var = tk.StringVar(value="Samples: (none)")
        self.preview_label_var = tk.StringVar(value="Preview: (none)")

        ttk.Label(status, textvariable=self.file_label_var).pack(anchor="w", padx=10, pady=(8, 2))
        ttk.Label(status, textvariable=self.samples_label_var).pack(anchor="w", padx=10, pady=2)
        ttk.Label(status, textvariable=self.preview_label_var).pack(anchor="w", padx=10, pady=(2, 8))

        # ===== Plots (Week 7-8) =====
        plots = ttk.LabelFrame(self, text="Plots (Week 7-8)")
        plots.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Tabs: Raw + FFT
        self.plot_tabs = ttk.Notebook(plots)
        self.plot_tabs.pack(fill="both", expand=True)

        self.tab_raw = ttk.Frame(self.plot_tabs)
        self.tab_fft = ttk.Frame(self.plot_tabs)
        self.plot_tabs.add(self.tab_raw, text="Raw Signal")
        self.plot_tabs.add(self.tab_fft, text="FFT")

        # --- Raw figure ---
        self.fig_raw = Figure(figsize=(7, 3.5), dpi=100)
        self.ax_raw = self.fig_raw.add_subplot(111)
        self.ax_raw.set_title("Time Domain")
        self.ax_raw.set_xlabel("t (s) or sample index")
        self.ax_raw.set_ylabel("Amplitude")

        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=self.tab_raw)
        self.canvas_raw.draw()
        self.canvas_raw.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar_raw = NavigationToolbar2Tk(self.canvas_raw, self.tab_raw)
        self.toolbar_raw.update()

        # --- FFT figure ---
        self.fig_fft = Figure(figsize=(7, 3.5), dpi=100)
        self.ax_fft = self.fig_fft.add_subplot(111)
        self.ax_fft.set_title("FFT Magnitude")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("|X(f)|")

        self.canvas_fft = FigureCanvasTkAgg(self.fig_fft, master=self.tab_fft)
        self.canvas_fft.draw()
        self.canvas_fft.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar_fft = NavigationToolbar2Tk(self.canvas_fft, self.tab_fft)
        self.toolbar_fft.update()

        # Initial empty plots
        self.update_plots()

    def on_load_csv(self):
        path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not path:
            return

        try:
            data = load_csv_numeric(path)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load CSV:\n{e}")
            return

        self.data = data
        self.loaded_path = path

        # Update status
        fname = os.path.basename(path)
        self.file_label_var.set(f"File: {fname}")

        # ===== Auto-select signal type based on filename =====
        name = fname.lower()
        if "ecg" in name:
            self.signal_type.set("ECG")
        elif "resp" in name:
            self.signal_type.set("Respiration")
        elif "temp" in name:
            self.signal_type.set("Temperature")
        elif "motion" in name or "imu" in name:
            self.signal_type.set("Motion")
        # ====================================================

        n = len(data.x)
        has_t = data.t is not None and len(data.t) == n
        self.samples_label_var.set(f"Samples: {n}   (time column: {'yes' if has_t else 'no'})")

        first = data.x[0]
        last = data.x[-1]
        self.preview_label_var.set(f"Preview: x[0]={first:.4f}   x[-1]={last:.4f}")

        # Update plots
        self.update_plots()

    def on_apply(self):
        if self.data is None:
            messagebox.showwarning("No Data", "Load a CSV file first.")
            return

        stype = self.signal_type.get()
        messagebox.showinfo("Week Check", f"Loaded data OK.\nSelected signal type: {stype}")

    def on_reset(self):
        self.data = None
        self.loaded_path = None
        self.file_label_var.set("File: (none)")
        self.samples_label_var.set("Samples: (none)")
        self.preview_label_var.set("Preview: (none)")
        self.update_plots()

    def update_plots(self):
        # Clear axes
        self.ax_raw.clear()
        self.ax_fft.clear()

        if self.data is None:
            self.ax_raw.set_title("Time Domain")
            self.ax_raw.text(
                0.5, 0.5, "Load a CSV to plot",
                ha="center", va="center",
                transform=self.ax_raw.transAxes
            )
            self.ax_raw.set_xlabel("t (s) or sample index")
            self.ax_raw.set_ylabel("Amplitude")
            self.ax_raw.grid(True)

            self.ax_fft.set_title("FFT Magnitude")
            self.ax_fft.text(
                0.5, 0.5, "Load a CSV to plot",
                ha="center", va="center",
                transform=self.ax_fft.transAxes
            )
            self.ax_fft.set_xlabel("Frequency (Hz)")
            self.ax_fft.set_ylabel("|X(f)|")
            self.ax_fft.grid(True)

            self.canvas_raw.draw()
            self.canvas_fft.draw()
            return

        # Build axes
        t_axis = make_time_axis(self.data.t, self.data.x)
        x = np.array(self.data.x, dtype=float)

        # Raw plot
        self.ax_raw.plot(t_axis, x)
        self.ax_raw.set_title("Time Domain")
        self.ax_raw.set_xlabel("t (s) or sample index")
        self.ax_raw.set_ylabel("Amplitude")
        self.ax_raw.grid(True)

        # FFT plot
        freqs, mag = compute_fft(self.data.x, t_axis)
        if freqs.size > 0:
            self.ax_fft.plot(freqs, mag)

        self.ax_fft.set_title("FFT Magnitude")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("|X(f)|")
        self.ax_fft.grid(True)

        self.canvas_raw.draw()
        self.canvas_fft.draw()