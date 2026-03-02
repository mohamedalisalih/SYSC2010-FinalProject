import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from io_loader import load_csv_numeric, SignalData


class SignalAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SYSC2010 Signal Analyzer")
        self.geometry("900x600")

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

        # (Filter controls will be used Week 8-9; keep placeholders)
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

        # ===== Status Frame (Week 6 deliverable proof) =====
        status = ttk.LabelFrame(self, text="Loaded Data Status")
        status.pack(fill="x", padx=10, pady=(0, 10))

        self.file_label_var = tk.StringVar(value="File: (none)")
        self.samples_label_var = tk.StringVar(value="Samples: (none)")
        self.preview_label_var = tk.StringVar(value="Preview: (none)")

        ttk.Label(status, textvariable=self.file_label_var).pack(anchor="w", padx=10, pady=(8, 2))
        ttk.Label(status, textvariable=self.samples_label_var).pack(anchor="w", padx=10, pady=2)
        ttk.Label(status, textvariable=self.preview_label_var).pack(anchor="w", padx=10, pady=(2, 8))

        # ===== Placeholder Frames (plots + stats later) =====
        plots = ttk.LabelFrame(self, text="Plots (Week 7-8)")
        plots.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        ttk.Label(plots, text="Next: embed Matplotlib plots here (raw + FFT).").pack(pady=10)

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

        # Update status display
        fname = os.path.basename(path)
        self.file_label_var.set(f"File: {fname}")

        n = len(data.x)
        has_t = data.t is not None and len(data.t) == n
        self.samples_label_var.set(f"Samples: {n}   (time column: {'yes' if has_t else 'no'})")

        # Preview
        first = data.x[0]
        last = data.x[-1]
        self.preview_label_var.set(f"Preview: x[0]={first:.4f}   x[-1]={last:.4f}")

    def on_apply(self):
        # Week 6: just validate that data exists and signal type is chosen
        if self.data is None:
            messagebox.showwarning("No Data", "Load a CSV file first.")
            return

        stype = self.signal_type.get()
        messagebox.showinfo("Week 6 Check", f"Loaded data OK.\nSelected signal type: {stype}")

    def on_reset(self):
        self.data = None
        self.loaded_path = None
        self.file_label_var.set("File: (none)")
        self.samples_label_var.set("Samples: (none)")
        self.preview_label_var.set("Preview: (none)")