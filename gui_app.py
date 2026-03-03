import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from io_loader import load_csv_numeric, SignalData
from analysis import (
    make_time_axis,
    compute_fft,
    compute_basic_metrics,
    estimate_hr_ecg,
    estimate_resp_rate,
    estimate_temp_slope,
)
from processing import FilterSpec, design_and_apply


class SignalAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SYSC2010 Signal Analyzer")
        self.geometry("1100x860")

        # Stored data
        self.data: SignalData | None = None
        self.loaded_path: str | None = None

        # Filtered signal storage
        self.filtered_x = None  # numpy array or None

        # ===== Controls Frame =====
        controls = ttk.LabelFrame(self, text="Controls")
        controls.pack(fill="x", padx=10, pady=10)

        # Row 0
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

        # Row 1 (filter params)
        ttk.Label(controls, text="Order:").grid(row=1, column=1, padx=8, pady=6, sticky="e")
        self.order_var = tk.StringVar(value="51")
        ttk.Entry(controls, textvariable=self.order_var, width=8).grid(row=1, column=2, padx=8, pady=6, sticky="w")

        ttk.Label(controls, text="Cutoff1 (Hz):").grid(row=1, column=3, padx=8, pady=6, sticky="e")
        self.cut1_var = tk.StringVar(value="15")
        ttk.Entry(controls, textvariable=self.cut1_var, width=10).grid(row=1, column=4, padx=8, pady=6, sticky="w")

        ttk.Label(controls, text="Cutoff2 (Hz):").grid(row=1, column=5, padx=8, pady=6, sticky="e")
        self.cut2_var = tk.StringVar(value="40")
        ttk.Entry(controls, textvariable=self.cut2_var, width=10).grid(row=1, column=6, padx=8, pady=6, sticky="w")

        ttk.Label(controls, text="fs (Hz):").grid(row=1, column=7, padx=8, pady=6, sticky="e")
        self.fs_var = tk.StringVar(value="100")
        ttk.Entry(controls, textvariable=self.fs_var, width=10).grid(row=1, column=8, padx=8, pady=6, sticky="w")

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

        # ===== Stats / Features =====
        stats = ttk.LabelFrame(self, text="Statistics / Features")
        stats.pack(fill="x", padx=10, pady=(0, 10))

        self.stats_raw_var = tk.StringVar(value="Raw: (load a CSV)")
        self.stats_proc_var = tk.StringVar(value="Processed: (load a CSV)")
        self.feature_var = tk.StringVar(value="Feature: (load a CSV)")

        ttk.Label(stats, textvariable=self.stats_raw_var).pack(anchor="w", padx=10, pady=(8, 2))
        ttk.Label(stats, textvariable=self.stats_proc_var).pack(anchor="w", padx=10, pady=2)
        ttk.Label(stats, textvariable=self.feature_var).pack(anchor="w", padx=10, pady=(2, 8))

        # ===== Plots =====
        plots = ttk.LabelFrame(self, text="Plots")
        plots.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.plot_tabs = ttk.Notebook(plots)
        self.plot_tabs.pack(fill="both", expand=True)

        self.tab_raw = ttk.Frame(self.plot_tabs)
        self.tab_fft = ttk.Frame(self.plot_tabs)
        self.plot_tabs.add(self.tab_raw, text="Raw / Filtered")
        self.plot_tabs.add(self.tab_fft, text="FFT")

        # Raw figure
        self.fig_raw = Figure(figsize=(7, 3.5), dpi=100)
        self.ax_raw = self.fig_raw.add_subplot(111)

        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=self.tab_raw)
        self.canvas_raw.draw()
        self.canvas_raw.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar_raw = NavigationToolbar2Tk(self.canvas_raw, self.tab_raw)
        self.toolbar_raw.update()

        # FFT figure
        self.fig_fft = Figure(figsize=(7, 3.5), dpi=100)
        self.ax_fft = self.fig_fft.add_subplot(111)

        self.canvas_fft = FigureCanvasTkAgg(self.fig_fft, master=self.tab_fft)
        self.canvas_fft.draw()
        self.canvas_fft.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar_fft = NavigationToolbar2Tk(self.canvas_fft, self.tab_fft)
        self.toolbar_fft.update()

        # Initial plots
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
        self.filtered_x = None

        # Status
        fname = os.path.basename(path)
        self.file_label_var.set(f"File: {fname}")

        # Auto-select signal type based on filename
        name = fname.lower()
        if "ecg" in name:
            self.signal_type.set("ECG")
        elif "resp" in name:
            self.signal_type.set("Respiration")
        elif "temp" in name:
            self.signal_type.set("Temperature")
        elif "motion" in name or "imu" in name:
            self.signal_type.set("Motion")

        n = len(data.x)
        has_t = data.t is not None and len(data.t) == n
        self.samples_label_var.set(f"Samples: {n}   (time column: {'yes' if has_t else 'no'})")

        self.preview_label_var.set(f"Preview: x[0]={data.x[0]:.4f}   x[-1]={data.x[-1]:.4f}")

        self.update_plots()

    def on_apply(self):
        if self.data is None:
            messagebox.showwarning("No Data", "Load a CSV file first.")
            return

        try:
            fs = float(self.fs_var.get())
            order = int(float(self.order_var.get()))
            cut1 = float(self.cut1_var.get())
            cut2 = float(self.cut2_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "Order, cutoffs, and fs must be numbers.")
            return

        ftype = self.filter_type.get()
        method = self.method.get()

        if fs <= 0:
            messagebox.showerror("Input Error", "fs must be > 0.")
            return

        nyq = fs / 2.0

        if ftype != "None":
            if cut1 <= 0 or cut1 >= nyq:
                messagebox.showerror("Input Error", f"Cutoff1 must be between 0 and {nyq:.2f}.")
                return

            if ftype == "BPF":
                if cut2 <= 0 or cut2 >= nyq:
                    messagebox.showerror("Input Error", f"Cutoff2 must be between 0 and {nyq:.2f}.")
                    return
                if cut2 <= cut1:
                    messagebox.showerror("Input Error", "For BPF, cutoff2 must be greater than cutoff1.")
                    return

        x = np.array(self.data.x, dtype=float)

        try:
            spec = FilterSpec(
                filter_type=ftype,
                method=method,
                order=order,
                cutoff1=cut1,
                cutoff2=cut2,
                fs=fs
            )
            self.filtered_x = design_and_apply(x, spec)
        except Exception as e:
            messagebox.showerror("Filter Error", str(e))
            return

        self.update_plots()

    def on_reset(self):
        self.data = None
        self.loaded_path = None
        self.filtered_x = None

        self.file_label_var.set("File: (none)")
        self.samples_label_var.set("Samples: (none)")
        self.preview_label_var.set("Preview: (none)")

        self.stats_raw_var.set("Raw: (load a CSV)")
        self.stats_proc_var.set("Processed: (load a CSV)")
        self.feature_var.set("Feature: (load a CSV)")

        self.update_plots()

    def update_stats_panel(self, t_axis: np.ndarray, x_raw: np.ndarray, x_proc: np.ndarray):
        m_raw = compute_basic_metrics(x_raw)
        m_proc = compute_basic_metrics(x_proc)

        self.stats_raw_var.set(
            f"Raw: mean={m_raw['mean']:.4f}, std={m_raw['std']:.4f}, rms={m_raw['rms']:.4f}, p2p={m_raw['ptp']:.4f}"
        )
        self.stats_proc_var.set(
            f"Processed: mean={m_proc['mean']:.4f}, std={m_proc['std']:.4f}, rms={m_proc['rms']:.4f}, p2p={m_proc['ptp']:.4f}"
        )

        # fs hint from GUI if available
        fs_hint = None
        try:
            fs_hint = float(self.fs_var.get())
        except Exception:
            fs_hint = None

        stype = self.signal_type.get()
        if stype == "ECG":
            hr = estimate_hr_ecg(t_axis, x_proc, fs_hint=fs_hint)
            self.feature_var.set("Feature: HR = (not enough peaks)" if hr is None else f"Feature: HR ≈ {hr:.1f} bpm")
        elif stype == "Respiration":
            br = estimate_resp_rate(t_axis, x_proc, fs_hint=fs_hint)
            self.feature_var.set(
                "Feature: Resp rate = (not enough peaks)" if br is None else f"Feature: Resp rate ≈ {br:.1f} breaths/min"
            )
        elif stype == "Temperature":
            slope = estimate_temp_slope(t_axis, x_proc)
            self.feature_var.set(
                "Feature: Temp slope = (need more points)" if slope is None else f"Feature: Trend slope ≈ {slope:.6f} units/s"
            )
        else:
            self.feature_var.set("Feature: (no feature for Motion yet)")

    def update_plots(self):
        self.ax_raw.clear()
        self.ax_fft.clear()

        if self.data is None:
            self.ax_raw.set_title("Raw / Filtered (Time Domain)")
            self.ax_raw.text(0.5, 0.5, "Load a CSV to plot", ha="center", va="center", transform=self.ax_raw.transAxes)
            self.ax_raw.set_xlabel("t (s) or sample index")
            self.ax_raw.set_ylabel("Amplitude")
            self.ax_raw.grid(True)

            self.ax_fft.set_title("FFT Magnitude")
            self.ax_fft.text(0.5, 0.5, "Load a CSV to plot", ha="center", va="center", transform=self.ax_fft.transAxes)
            self.ax_fft.set_xlabel("Frequency (Hz)")
            self.ax_fft.set_ylabel("|X(f)|")
            self.ax_fft.grid(True)

            self.canvas_raw.draw()
            self.canvas_fft.draw()
            return

        # Time axis + raw data
        t_axis = make_time_axis(self.data.t, self.data.x)
        x = np.array(self.data.x, dtype=float)

        # Processed signal (filtered if present)
        x_proc = x
        if self.filtered_x is not None and len(self.filtered_x) == len(x):
            x_proc = np.array(self.filtered_x, dtype=float)

        # Plot raw + processed
        self.ax_raw.plot(t_axis, x, label="Raw")
        if x_proc is not x:
            self.ax_raw.plot(t_axis, x_proc, linestyle="--", label="Processed")

        self.ax_raw.set_title("Raw / Processed (Time Domain)")
        self.ax_raw.set_xlabel("t (s) or sample index")
        self.ax_raw.set_ylabel("Amplitude")
        self.ax_raw.grid(True)
        self.ax_raw.legend()

        # FFT uses processed if available
        freqs, mag = compute_fft(list(x_proc), np.array(t_axis, dtype=float))
        if freqs.size > 0:
            self.ax_fft.plot(freqs, mag)

        self.ax_fft.set_title("FFT Magnitude")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("|X(f)|")
        self.ax_fft.grid(True)

        # Update stats panel
        self.update_stats_panel(np.array(t_axis, dtype=float), x, x_proc)

        self.canvas_raw.draw()
        self.canvas_fft.draw()