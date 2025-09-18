import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

from .halstead_fs import analyze_fsharp_source, HalsteadResult


class HalsteadApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Halstead Metrics for F#")
        self.geometry("980x720")
        self._create_widgets()

    def _create_widgets(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        open_btn = ttk.Button(toolbar, text="Open F# File", command=self._open_file)
        open_btn.pack(side=tk.LEFT, padx=6, pady=6)

        analyze_btn = ttk.Button(toolbar, text="Analyze", command=self._analyze)
        analyze_btn.pack(side=tk.LEFT, padx=6, pady=6)

        self.status_var = tk.StringVar(value="Open a .fs or .fsx file and click Analyze")
        status = ttk.Label(toolbar, textvariable=self.status_var)
        status.pack(side=tk.LEFT, padx=12)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: source editor
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        self.text = tk.Text(left, wrap=tk.NONE, undo=True)
        self.text.pack(fill=tk.BOTH, expand=True)

        # Right: tabs with results
        right = ttk.Notebook(paned)
        paned.add(right, weight=1)

        self.operators_tree = self._create_table(right, ("Operator", "Count"), "Operators")
        self.operands_tree = self._create_table(right, ("Operand", "Count"), "Operands")
        self.metrics_tree = self._create_table(right, ("Metric", "Value"), "Metrics")

    def _create_table(self, parent: ttk.Notebook, columns: tuple, title: str) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        parent.add(frame, text=title)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200, anchor=tk.W)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        return tree

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open F# source",
            filetypes=[("F# files", "*.fs *.fsx"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.text.delete("1.0", tk.END)
            self.text.insert("1.0", content)
            self.status_var.set(f"Loaded: {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _analyze(self) -> None:
        source = self.text.get("1.0", tk.END)
        try:
            result = analyze_fsharp_source(source)
            self._populate_results(result)
            self.status_var.set("Analysis complete")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _populate_results(self, res: HalsteadResult) -> None:
        for tree in (self.operators_tree, self.operands_tree, self.metrics_tree):
            for item in tree.get_children():
                tree.delete(item)

        # Fill operators and operands frequency tables
        for op, cnt in res.operator_frequencies:
            self.operators_tree.insert("", tk.END, values=(op, cnt))
        for opd, cnt in res.operand_frequencies:
            self.operands_tree.insert("", tk.END, values=(opd, cnt))

        # All metrics in one table (base + extended)
        all_metrics = [
            # Base metrics (6 base metrics as per req.md)
            ("η₁ (unique operators)", res.eta1_unique_operators),
            ("η₂ (unique operands)", res.eta2_unique_operands),
            ("N₁ (total operators)", res.N1_total_operators),
            ("N₂ (total operands)", res.N2_total_operands),
            ("η = η₁ + η₂ (vocabulary)", res.eta_vocabulary),
            ("N = N₁ + N₂ (length)", res.N_length),
            # Extended metrics (3 extended metrics as per req.md)
            ("V = N log₂ η (volume)", f"{res.V_volume:.3f}"),
        ]

        for name, val in all_metrics:
            self.metrics_tree.insert("", tk.END, values=(name, val))


def run_app() -> None:
    app = HalsteadApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()


