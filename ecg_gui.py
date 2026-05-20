import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from edt import ecg_to_csv
from edt_utils import plot_ecg_signal
import numpy as np


class ECGApp:

    def __init__(self, root):

        self.root = root
        self.root.title("ECG Digitizer")
        self.root.geometry("1400x900")

        # =========================================================
        # VARIABLES
        # =========================================================

        self.image_path    = tk.StringVar()
        self.template_path = tk.StringVar()

        # ── Basic ──────────────────────────────────────────────
        self.strategy    = tk.StringVar(value="none")
        self.thres_value = tk.IntVar(value=127)
        self.dilation    = tk.IntVar(value=10)
        self.layout      = tk.StringVar(value="(3,4)")
        self.pulse       = tk.StringVar(value="[0,1,2]")
        self.rhythm      = tk.IntVar(value=4)
        self.verbose     = tk.IntVar(value=0)
        self.show_images = tk.BooleanVar(value=False)

        # ── Advanced ───────────────────────────────────────────
        self.perc_space_leads = tk.DoubleVar(value=0.2)
        self.perc_max_dist    = tk.DoubleVar(value=0.7)
        self.kSize2d          = tk.IntVar(value=3)
        self.kSize1d          = tk.IntVar(value=3)
        self.lower_h          = tk.IntVar(value=0)
        self.lower_s          = tk.IntVar(value=0)
        self.lower_v          = tk.IntVar(value=0)
        self.upper_h          = tk.IntVar(value=179)
        self.upper_s          = tk.IntVar(value=255)
        self.upper_v          = tk.IntVar(value=220)
        self.sample_frequency = tk.IntVar(value=500)
        self.mmpsec           = tk.DoubleVar(value=25)
        self.mmpmv            = tk.DoubleVar(value=10)

        # Internal state
        self._debug_canvases = []   # keep references to avoid GC
        self._last_df        = None
        self._last_layout    = (3, 4)

        # =========================================================
        # NOTEBOOK  (3 tabs)
        # =========================================================

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.tab_config  = ttk.Frame(self.notebook)
        self.tab_result  = ttk.Frame(self.notebook)
        self.tab_debug   = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_config, text="⚙  Configuração")
        self.notebook.add(self.tab_result, text="📈  Resultado")
        self.notebook.add(self.tab_debug,  text="🔍  Debug / Verbose")

        self._build_config_tab()
        self._build_result_tab()
        self._build_debug_tab()

    # =========================================================
    # TAB: CONFIGURAÇÃO
    # =========================================================

    def _build_config_tab(self):

        # ── File selectors ────────────────────────────────────
        top_frame = ttk.Frame(self.tab_config)
        top_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(top_frame, text="Selecionar ECG",
                   command=self.load_image).grid(row=0, column=0, padx=5)
        ttk.Entry(top_frame, textvariable=self.image_path,
                  width=55).grid(row=0, column=1, padx=5)

        ttk.Button(top_frame, text="Selecionar Template",
                   command=self.load_template).grid(row=0, column=2, padx=5)
        ttk.Entry(top_frame, textvariable=self.template_path,
                  width=35).grid(row=0, column=3, padx=5)

        # ── Main split: left controls | right preview ─────────
        main_frame = ttk.Frame(self.tab_config)
        main_frame.pack(fill="both", expand=True)

        left_panel  = ttk.Frame(main_frame)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)

        # ── Basic config ──────────────────────────────────────
        basic_frame = ttk.LabelFrame(left_panel, text="Configuração Básica")
        basic_frame.pack(fill="x", pady=5)

        ttk.Label(basic_frame, text="Strategy").grid(row=0, column=0, sticky="w")
        ttk.Combobox(basic_frame, textvariable=self.strategy,
                     values=["none", "filter", "color"],
                     state="readonly", width=15
                     ).grid(row=0, column=1, padx=5, pady=4)

        ttk.Label(basic_frame, text="Threshold").grid(row=1, column=0, sticky="w")
        tk.Scale(basic_frame, from_=0, to=255, orient="horizontal",
                 variable=self.thres_value, length=200
                 ).grid(row=1, column=1)

        ttk.Label(basic_frame, text="Dilation").grid(row=2, column=0, sticky="w")
        tk.Scale(basic_frame, from_=0, to=20, orient="horizontal",
                 variable=self.dilation, length=200
                 ).grid(row=2, column=1)

        ttk.Label(basic_frame, text="Layout").grid(row=3, column=0, sticky="w")
        ttk.Combobox(basic_frame, textvariable=self.layout,
                     values=["(3,4)", "(12,1)"],
                     state="readonly", width=15
                     ).grid(row=3, column=1, padx=5, pady=4)

        ttk.Label(basic_frame, text="Pulse").grid(row=4, column=0, sticky="w")
        ttk.Combobox(basic_frame, textvariable=self.pulse,
                     values=["0", "-1", "[0,1,2]"],
                     state="readonly", width=15
                     ).grid(row=4, column=1, padx=5, pady=4)

        ttk.Label(basic_frame, text="Rhythm").grid(row=5, column=0, sticky="w")
        tk.Spinbox(basic_frame, from_=0, to=12,
                   textvariable=self.rhythm, width=10
                   ).grid(row=5, column=1, padx=5, pady=4)

        ttk.Label(basic_frame, text="Verbose (0-3)").grid(row=6, column=0, sticky="w", pady=5)
        tk.Spinbox(basic_frame, from_=0, to=3,
                   textvariable=self.verbose, width=5
                   ).grid(row=6, column=1, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(basic_frame, text="Show Images",
                        variable=self.show_images
                        ).grid(row=7, column=0, sticky="w", pady=5)

        # ── Advanced config ───────────────────────────────────
        adv_frame = ttk.LabelFrame(left_panel, text="Configuração Avançada")
        adv_frame.pack(fill="x", pady=10)

        def _row(parent, label, var, r):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w")
            ttk.Entry(parent, textvariable=var, width=10).grid(row=r, column=1, pady=2)

        _row(adv_frame, "perc_space_leads", self.perc_space_leads, 0)
        _row(adv_frame, "perc_max_dist",    self.perc_max_dist,    1)
        _row(adv_frame, "Kernel 2D",        self.kSize2d,          2)
        _row(adv_frame, "Kernel 1D",        self.kSize1d,          3)

        hsv_frame = ttk.LabelFrame(adv_frame, text="HSV Filter")
        hsv_frame.grid(row=4, column=0, columnspan=2, pady=8, sticky="ew")

        ttk.Label(hsv_frame, text="Lower (H S V)").grid(row=0, column=0, padx=4)
        ttk.Entry(hsv_frame, textvariable=self.lower_h, width=5).grid(row=0, column=1)
        ttk.Entry(hsv_frame, textvariable=self.lower_s, width=5).grid(row=0, column=2)
        ttk.Entry(hsv_frame, textvariable=self.lower_v, width=5).grid(row=0, column=3)

        ttk.Label(hsv_frame, text="Upper (H S V)").grid(row=1, column=0, padx=4)
        ttk.Entry(hsv_frame, textvariable=self.upper_h, width=5).grid(row=1, column=1)
        ttk.Entry(hsv_frame, textvariable=self.upper_s, width=5).grid(row=1, column=2)
        ttk.Entry(hsv_frame, textvariable=self.upper_v, width=5).grid(row=1, column=3)

        _row(adv_frame, "sample_frequency", self.sample_frequency, 5)
        _row(adv_frame, "mm/sec",           self.mmpsec,           6)
        _row(adv_frame, "mm/mV",            self.mmpmv,            7)

        # ── Process button ────────────────────────────────────
        ttk.Button(left_panel, text="▶  PROCESSAR ECG",
                   command=self.process
                   ).pack(fill="x", pady=12)

        # ── Image preview (right panel) ───────────────────────
        preview_frame = ttk.Frame(right_panel)
        preview_frame.pack(fill="x", pady=10, padx=5)

        # ECG image
        ecg_preview = ttk.LabelFrame(preview_frame, text="Imagem ECG")
        ecg_preview.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.image_label = ttk.Label(ecg_preview)
        self.image_label.pack(pady=5, padx=5)

        # Template image
        tpl_preview = ttk.LabelFrame(preview_frame, text="Template (Pulso)")
        tpl_preview.pack(side="left", fill="both", padx=(5, 0))
        self.template_label = ttk.Label(tpl_preview)
        self.template_label.pack(pady=5, padx=5)

    # =========================================================
    # TAB: RESULTADO
    # =========================================================

    def _build_result_tab(self):
        """Scrollable area that shows the reconstructed ECG in clinical layout."""

        toolbar_frame = ttk.Frame(self.tab_result)
        toolbar_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(toolbar_frame, text="💾  Salvar imagem",
                   command=self._save_result_image).pack(side="left", padx=4)

        ttk.Label(toolbar_frame,
                  text="ECG reconstruído — use a barra de ferramentas para zoom/pan"
                  ).pack(side="left", padx=10)

        # Scrollable canvas
        container = ttk.Frame(self.tab_result)
        container.pack(fill="both", expand=True)

        self._result_canvas_widget = tk.Canvas(container, bg="#f5f5f5")
        vsb = ttk.Scrollbar(container, orient="vertical",
                            command=self._result_canvas_widget.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal",
                            command=self._result_canvas_widget.xview)

        self._result_canvas_widget.configure(
            yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self._result_canvas_widget.pack(fill="both", expand=True)

        self._result_inner = ttk.Frame(self._result_canvas_widget)
        self._result_canvas_widget.create_window(
            (0, 0), window=self._result_inner, anchor="nw")
        self._result_inner.bind(
            "<Configure>",
            lambda e: self._result_canvas_widget.configure(
                scrollregion=self._result_canvas_widget.bbox("all")))

        self._result_fig_canvas = None   # will hold FigureCanvasTkAgg

    # =========================================================
    # TAB: DEBUG / VERBOSE
    # =========================================================

    def _build_debug_tab(self):
        """Scrollable area that receives all intermediate debug figures."""

        toolbar = ttk.Frame(self.tab_debug)
        toolbar.pack(fill="x", padx=5, pady=5)

        ttk.Button(toolbar, text="🗑  Limpar",
                   command=self.clear_debug).pack(side="left")
        ttk.Label(toolbar,
                  text="Plots de debug aparecem aqui quando Verbose está ativo."
                  ).pack(side="left", padx=10)

        container = ttk.Frame(self.tab_debug)
        container.pack(fill="both", expand=True)

        self._debug_tk_canvas = tk.Canvas(container, bg="#fafafa")
        vsb = ttk.Scrollbar(container, orient="vertical",
                            command=self._debug_tk_canvas.yview)

        self._debug_tk_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._debug_tk_canvas.pack(side="left", fill="both", expand=True)

        self._debug_inner = ttk.Frame(self._debug_tk_canvas)
        self._debug_tk_canvas.create_window(
            (0, 0), window=self._debug_inner, anchor="nw")
        self._debug_inner.bind(
            "<Configure>",
            lambda e: self._debug_tk_canvas.configure(
                scrollregion=self._debug_tk_canvas.bbox("all")))

        # Mouse scroll — cross-platform
        def _on_mousewheel(event):
            # Windows/macOS: event.delta; Linux: Button-4/5
            if event.num == 4:
                self._debug_tk_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self._debug_tk_canvas.yview_scroll(1, "units")
            else:
                self._debug_tk_canvas.yview_scroll(
                    int(-1 * (event.delta / 120)), "units")

        # Bind to canvas and inner frame so scroll works wherever the mouse is
        for widget in (self._debug_tk_canvas, self._debug_inner):
            widget.bind("<MouseWheel>", _on_mousewheel)   # Windows / macOS
            widget.bind("<Button-4>",   _on_mousewheel)   # Linux scroll up
            widget.bind("<Button-5>",   _on_mousewheel)   # Linux scroll down

        # Also propagate scroll from any child widget added later
        self._debug_scroll_handler = _on_mousewheel

        # Buffer that accumulates print() lines between figure calls
        self._pending_log_lines = []

    # =========================================================
    # DEBUG CALLBACK  (called by edt.py / edt_utils.py)
    # =========================================================

    def add_debug_figure(self, fig):
        """
        Receive a matplotlib Figure from the processing pipeline and render
        it inside the Debug tab, with the accumulated log lines on the right.
        Called only when verbose is active.
        """
        # Outer row frame
        row_frame = ttk.Frame(self._debug_inner, relief="solid", borderwidth=1)
        row_frame.pack(fill="x", pady=6, padx=6)

        # Left: matplotlib figure
        fig_frame = ttk.Frame(row_frame)
        fig_frame.pack(side="left", fill="both", expand=True)

        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._debug_canvases.append(canvas)   # prevent GC

        # Right: log text for messages printed before this figure
        log_frame = ttk.Frame(row_frame, width=340)
        log_frame.pack(side="left", fill="y", padx=(4, 4), pady=4)
        log_frame.pack_propagate(False)

        log_text = tk.Text(
            log_frame,
            wrap="word",
            bg="#1e1e1e",
            fg="#d4d4d4",
            font=("Courier", 12),
            relief="flat",
            state="normal",
        )
        log_text.pack(fill="both", expand=True)

        # Drain the pending log buffer into this widget
        content = "".join(self._pending_log_lines)
        log_text.insert("1.0", content if content.strip() else "(sem mensagens)")
        log_text.configure(state="disabled")
        self._pending_log_lines.clear()

        # Bind scroll to every new widget so mouse works anywhere on the row
        for w in (row_frame, fig_frame, canvas.get_tk_widget(), log_frame, log_text):
            w.bind("<MouseWheel>", self._debug_scroll_handler)
            w.bind("<Button-4>",   self._debug_scroll_handler)
            w.bind("<Button-5>",   self._debug_scroll_handler)

        # Scroll to the bottom
        self._debug_inner.update_idletasks()
        self._debug_tk_canvas.configure(
            scrollregion=self._debug_tk_canvas.bbox("all"))
        self._debug_tk_canvas.yview_moveto(1.0)

    def clear_debug(self):
        """Remove all figures from the debug tab."""
        for widget in self._debug_inner.winfo_children():
            widget.destroy()
        self._debug_canvases.clear()
        self._pending_log_lines.clear()

    # =========================================================
    # LOAD IMAGE / TEMPLATE
    # =========================================================

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not path:
            return
        self.image_path.set(path)
        img = Image.open(path)
        img.thumbnail((600, 380))
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.tk_img)

    def load_template(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not path:
            return
        self.template_path.set(path)
        img = Image.open(path)
        img.thumbnail((180, 180))
        self.tk_template_img = ImageTk.PhotoImage(img)
        self.template_label.configure(image=self.tk_template_img)

    # =========================================================
    # PROCESS
    # =========================================================

    def process(self):
        try:
            layout = eval(self.layout.get())
            pulse  = eval(self.pulse.get())

            lower = (self.lower_h.get(), self.lower_s.get(), self.lower_v.get())
            upper = (self.upper_h.get(), self.upper_s.get(), self.upper_v.get())

            pulse_width_mm  = 5
            pulse_height_mm = 10
            pulse_per_sec   = pulse_width_mm  / self.mmpsec.get()
            pulse_per_mv    = pulse_height_mm / self.mmpmv.get()
            time_lead       = 2.5
            num_sampling_points = int(time_lead * self.sample_frequency.get())

            verbose_active = self.verbose.get() > 0

            # Clear previous debug figures before a new run
            if verbose_active:
                self.clear_debug()

            # Redirect stdout → pending log buffer (GUI mode)
            import sys
            _orig_stdout = sys.stdout

            class _LogBuffer:
                def __init__(self, lines):
                    self._lines = lines
                def write(self, msg):
                    if msg:
                        self._lines.append(msg)
                def flush(self):
                    pass

            if verbose_active:
                sys.stdout = _LogBuffer(self._pending_log_lines)

            config_dict = {
                # ECG structure
                'layout':          layout,
                'pulse':           pulse,
                'rhythm':          self.rhythm.get(),
                # Debug
                'verbose':         self.verbose.get(),
                'show_images':     self.show_images.get(),
                'plot_callback':   self.add_debug_figure if verbose_active else None,
                # Processing
                'strategy':        self.strategy.get(),
                'thres_value':     self.thres_value.get(),
                'lower':           lower,
                'upper':           upper,
                'kSized2d':        self.kSize2d.get(),
                'kSized1d':        self.kSize1d.get(),
                'perc_space_leads':self.perc_space_leads.get(),
                'dilation':        self.dilation.get(),
                'perc_max_dist':   self.perc_max_dist.get(),
                # Physiology
                'mmpsec':          self.mmpsec.get(),
                'mmpmv':           self.mmpmv.get(),
                'pulse_width_mm':  pulse_width_mm,
                'pulse_height_mm': pulse_height_mm,
                'pulse_per_sec':   pulse_per_sec,
                'pulse_per_mv':    pulse_per_mv,
                'sample_frequency':self.sample_frequency.get(),
                'time_lead':       time_lead,
                'num_sampling_points': num_sampling_points,
                'location':        'right',
            }

            result = ecg_to_csv(
                self.image_path.get(),
                self.template_path.get(),
                "output.csv",
                config_dict
            )

            df = result["df"]
            self._last_df     = df
            self._last_layout = layout

            # Render the result ECG
            self._plot_result(df, layout)

            # Switch to result tab
            self.notebook.select(self.tab_result)

            # If verbose, also switch to debug tab to show intermediate plots
            if verbose_active:
                self.notebook.select(self.tab_debug)

            messagebox.showinfo("Sucesso", "ECG processado com sucesso.")

        except Exception as e:
            sys.stdout = _orig_stdout
            messagebox.showerror("Erro", str(e))
            raise   # re-raise so the full traceback appears in the terminal

        finally:
            sys.stdout = _orig_stdout
            # Flush any remaining log lines that came after the last figure
            if verbose_active and self._pending_log_lines:
                row_frame = ttk.Frame(self._debug_inner, relief="solid", borderwidth=1)
                row_frame.pack(fill="x", pady=6, padx=6)
                log_text = tk.Text(
                    row_frame, wrap="word", bg="#1e1e1e", fg="#d4d4d4",
                    font=("Courier", 12), relief="flat", state="normal",
                )
                log_text.pack(fill="both", expand=True, padx=4, pady=4)
                log_text.insert("1.0", "".join(self._pending_log_lines))
                log_text.configure(state="disabled")
                self._pending_log_lines.clear()
                self._debug_inner.update_idletasks()
                self._debug_tk_canvas.configure(
                    scrollregion=self._debug_tk_canvas.bbox("all"))

    # =========================================================
    # PLOT RESULT  (clinical grid layout inside Result tab)
    # =========================================================

    def _plot_result(self, df, layout):
        """Render the reconstructed ECG in a clinical grid inside the Result tab."""

        # Destroy previous figure if any
        if self._result_fig_canvas is not None:
            self._result_fig_canvas.get_tk_widget().destroy()
            plt.close("all")

        n_rows, n_cols = layout
        fs = self.sample_frequency.get()

        # Compute figure height dynamically so each row has enough space
        fig_w = max(14, n_cols * 3.5)
        fig_h = max(6,  n_rows * 2.5)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
        fig.suptitle("ECG Reconstruído", fontsize=14)

        columns = list(df.columns)

        for index, col in enumerate(columns):
            if n_rows == 1 and n_cols == 1:
                ax = axes
            elif n_rows == 1 or n_cols == 1:
                ax = axes[index]
            else:
                row_idx = index // n_cols
                col_idx = index  % n_cols
                ax = axes[row_idx][col_idx]

            signal = df[col]
            ts     = np.arange(signal.size) / fs
            plot_ecg_signal(ts, signal, ax)
            ax.set_title(col, fontsize=9)

        fig.subplots_adjust(top=0.92, hspace=0.5, wspace=0.4)

        self._result_fig_canvas = FigureCanvasTkAgg(fig, master=self._result_inner)
        self._result_fig_canvas.draw()
        self._result_fig_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Update scroll region
        self._result_inner.update_idletasks()
        self._result_canvas_widget.configure(
            scrollregion=self._result_canvas_widget.bbox("all"))

    # =========================================================
    # SAVE RESULT IMAGE
    # =========================================================

    def _save_result_image(self):
        if self._last_df is None:
            messagebox.showwarning("Aviso", "Nenhum ECG processado ainda.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if not path:
            return
        # Re-generate and save
        n_rows, n_cols = self._last_layout
        fs = self.sample_frequency.get()
        fig_w = max(14, n_cols * 3.5)
        fig_h = max(6,  n_rows * 2.5)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
        fig.suptitle("ECG Reconstruído", fontsize=14)
        columns = list(self._last_df.columns)
        for index, col in enumerate(columns):
            if n_rows == 1 and n_cols == 1:
                ax = axes
            elif n_rows == 1 or n_cols == 1:
                ax = axes[index]
            else:
                ax = axes[index // n_cols][index % n_cols]
            signal = self._last_df[col]
            ts     = np.arange(signal.size) / fs
            plot_ecg_signal(ts, signal, ax)
            ax.set_title(col, fontsize=9)
        fig.subplots_adjust(top=0.92, hspace=0.5, wspace=0.4)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        messagebox.showinfo("Salvo", f"Imagem salva em:\n{path}")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app  = ECGApp(root)
    root.mainloop()
