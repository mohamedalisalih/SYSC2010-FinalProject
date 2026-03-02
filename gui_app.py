import tkinter as tk
from tkinter import ttk


class SignalAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SYSC2010 Signal Analyzer")
        self.geometry("900x600")

        # --- Controls frame ---
        controls = ttk.LabelFrame(self, text="Controls")
        controls.pack(fill="x", padx=10, pady=10)

        ttk.Button(controls, text="Load CSV").grid(row=0, column=0, padx=8, pady=8, sticky="w")

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

        ttk.Button(controls, text="Apply").grid(row=0, column=7, padx=8, pady=8, sticky="w")
        ttk.Button(controls, text="Reset").grid(row=0, column=8, padx=8, pady=8, sticky="w")

        # --- Placeholder frames (plots + stats) ---
        plots = ttk.LabelFrame(self, text="Plots (coming next)")
        plots.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        ttk.Label(plots, text="Time-domain plot will go here").pack(pady=10)
        ttk.Label(plots, text="FFT plot will go here").pack(pady=10)

        stats = ttk.LabelFrame(self, text="Stats / Features (coming next)")
        stats.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(stats, text="Mean: ___   Std: ___   RMS: ___   Peak-to-Peak: ___   Feature: ___").pack(padx=10, pady=10)