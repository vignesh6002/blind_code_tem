"""
feeder.py
=========
Browse and select a folder of images → feed them through nikhil_mains
(YOLO + MiDaS) → auto-save annotated outputs to ./calibration_images/

Run:
    python feeder.py

Requirements: same as nikhil_mains.py  (ultralytics, torch, cv2, flask …)
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR    = Path(__file__).parent.resolve()
OUTPUT_DIR     = str(PROJECT_DIR / "calibration_images")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Load nikhil_mains once (models load at import time)
# ---------------------------------------------------------------------------
print("[Feeder] Loading models from nikhil_mains … (this may take ~30 s)")
sys.path.insert(0, str(PROJECT_DIR))
import nikhil_mains as nm
print("[Feeder] Models ready.\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def collect_images(folder: str) -> list[str]:
    """Return sorted list of supported image paths inside folder."""
    return sorted(
        str(p) for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class FeederApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calibration Image Feeder")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")

        self._selected_folder = tk.StringVar(value="No folder selected")
        self._image_paths: list[str] = []
        self._running = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        PAD = 14
        FG  = "#e0e0e0"
        ACC = "#00ff99"
        BG  = "#1a1a2e"
        BTN = "#16213e"

        # Title
        tk.Label(self, text="Calibration Image Feeder",
                 font=("Consolas", 16, "bold"), fg=ACC, bg=BG
                 ).pack(pady=(PAD, 4))

        tk.Label(self, text="Select a folder of images → feed through YOLO + MiDaS\n"
                            "→ outputs saved to  ./calibration_images/",
                 font=("Consolas", 9), fg="#888888", bg=BG, justify="center"
                 ).pack(pady=(0, PAD))

        # Folder row
        folder_frame = tk.Frame(self, bg=BG)
        folder_frame.pack(fill="x", padx=PAD, pady=4)

        tk.Label(folder_frame, textvariable=self._selected_folder,
                 font=("Consolas", 9), fg=FG, bg=BTN,
                 anchor="w", relief="sunken", width=52, padx=6
                 ).pack(side="left", fill="x", expand=True)

        tk.Button(folder_frame, text="📂  Browse",
                  font=("Consolas", 9, "bold"), fg="#0d0d0d", bg=ACC,
                  activebackground="#00cc77", relief="flat", padx=8,
                  command=self._browse
                  ).pack(side="left", padx=(8, 0))

        # Image count label
        self._count_label = tk.Label(self, text="",
                                     font=("Consolas", 9), fg="#aaaaaa", bg=BG)
        self._count_label.pack()

        # Output dir display
        tk.Label(self,
                 text=f"Output → {OUTPUT_DIR}",
                 font=("Consolas", 8), fg="#555577", bg=BG
                 ).pack(pady=(0, 6))

        # Progress bar
        self._progress = ttk.Progressbar(self, orient="horizontal",
                                         length=460, mode="determinate")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=BTN,
                        background=ACC, thickness=18)
        self._progress.pack(padx=PAD, pady=6)

        # Progress text
        self._prog_label = tk.Label(self, text="",
                                    font=("Consolas", 9), fg=FG, bg=BG)
        self._prog_label.pack()

        # Log box
        log_frame = tk.Frame(self, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=PAD, pady=(4, 0))

        self._log = tk.Text(log_frame, height=10, width=62,
                            bg="#0d0d0d", fg="#aaffaa",
                            font=("Consolas", 8), state="disabled",
                            relief="flat", insertbackground=ACC)
        sb = tk.Scrollbar(log_frame, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        self._log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Run / Cancel buttons
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(pady=PAD)

        self._run_btn = tk.Button(btn_frame, text="▶  Run",
                                  font=("Consolas", 11, "bold"),
                                  fg="#0d0d0d", bg=ACC,
                                  activebackground="#00cc77",
                                  relief="flat", padx=20, pady=6,
                                  command=self._start, state="disabled")
        self._run_btn.pack(side="left", padx=6)

        self._cancel_btn = tk.Button(btn_frame, text="⏹  Cancel",
                                     font=("Consolas", 11, "bold"),
                                     fg="#0d0d0d", bg="#ff4444",
                                     activebackground="#cc0000",
                                     relief="flat", padx=20, pady=6,
                                     command=self._cancel, state="disabled")
        self._cancel_btn.pack(side="left", padx=6)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _browse(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        self._selected_folder.set(folder)
        self._image_paths = collect_images(folder)
        n = len(self._image_paths)
        if n == 0:
            self._count_label.config(text="⚠  No supported images found in that folder.",
                                     fg="#ff4444")
            self._run_btn.config(state="disabled")
        else:
            self._count_label.config(text=f"✔  {n} image(s) found", fg="#00ff99")
            self._run_btn.config(state="normal")
        self._progress["value"] = 0
        self._prog_label.config(text="")
        self._log_clear()

    def _start(self):
        if not self._image_paths:
            messagebox.showwarning("No images", "Please select a folder first.")
            return
        self._running     = True
        self._cancel_flag = False
        self._run_btn.config(state="disabled")
        self._cancel_btn.config(state="normal")
        self._progress["maximum"] = len(self._image_paths)
        self._progress["value"]   = 0
        self._log_clear()
        self._log_write(f"Starting … {len(self._image_paths)} images → {OUTPUT_DIR}\n")

        # Run in background thread so the GUI stays responsive
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _cancel(self):
        self._cancel_flag = True
        self._log_write("\n[Feeder] Cancellation requested — finishing current image …\n")

    # ------------------------------------------------------------------
    # Pipeline (background thread)
    # ------------------------------------------------------------------
    def _run_pipeline(self):
        paths = self._image_paths[:]  # snapshot

        def on_progress(done, total):
            if self._cancel_flag:
                raise InterruptedError("User cancelled")
            self.after(0, self._update_progress, done, total)

        try:
            saved = nm.process_images(paths, OUTPUT_DIR,
                                      progress_callback=on_progress)
            self.after(0, self._on_done, saved)
        except InterruptedError:
            self.after(0, self._on_cancelled)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    # ------------------------------------------------------------------
    # UI update callbacks (called from main thread via .after())
    # ------------------------------------------------------------------
    def _update_progress(self, done, total):
        self._progress["value"] = done
        self._prog_label.config(text=f"{done} / {total} images processed")
        self._log_write(f"  [{done:>3}/{total}] done\n")

    def _on_done(self, saved):
        self._running = False
        self._run_btn.config(state="normal")
        self._cancel_btn.config(state="disabled")
        msg = f"\n✔  Finished! {len(saved)} annotated images saved to:\n   {OUTPUT_DIR}"
        self._log_write(msg + "\n")
        messagebox.showinfo("Done", msg)

    def _on_cancelled(self):
        self._running = False
        self._run_btn.config(state="normal")
        self._cancel_btn.config(state="disabled")
        self._log_write("\n⏹  Cancelled by user.\n")

    def _on_error(self, err_msg):
        self._running = False
        self._run_btn.config(state="normal")
        self._cancel_btn.config(state="disabled")
        self._log_write(f"\n❌  ERROR: {err_msg}\n")
        messagebox.showerror("Error", err_msg)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------
    def _log_write(self, text: str):
        self._log.config(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.config(state="disabled")

    def _log_clear(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = FeederApp()
    app.mainloop()
