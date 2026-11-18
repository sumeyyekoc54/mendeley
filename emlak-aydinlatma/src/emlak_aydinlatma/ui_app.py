"""Türkçe Tkinter arayüzü — emlak fotoğraf aydınlatma MVP."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .io_exif import default_output_dir, heif_available
from .looks import LOOK_LABELS_TR, LookProfile, apply_look
from .pipeline import EnhanceResult, EnhanceSettings, Mode, process_file, save_result


MAX_PHOTOS = 3
PREVIEW_MAX = 640


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb, mode="RGB")


def _fit(img: Image.Image, max_side: int = PREVIEW_MAX) -> Image.Image:
    w, h = img.size
    scale = min(max_side / max(w, 1), max_side / max(h, 1), 1.0)
    if scale >= 0.999:
        return img
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _label_strip(img: Image.Image, text: str) -> Image.Image:
    """Add a small caption bar above the image."""
    bar_h = 22
    out = Image.new("RGB", (img.width, img.height + bar_h), (32, 32, 32))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    draw.text((6, 4), text, fill=(235, 235, 235))
    return out


class PhotoSlot:
    def __init__(self, path: Path):
        self.path = path
        self.settings = EnhanceSettings()
        self.result: EnhanceResult | None = None


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Emlak Fotoğraf Aydınlatma")
        self.geometry("1180x780")
        self.minsize(960, 640)

        self.slots: list[PhotoSlot] = []
        self.active_index = 0
        self._photo_left: ImageTk.PhotoImage | None = None
        self._photo_right: ImageTk.PhotoImage | None = None
        self._photo_blend: ImageTk.PhotoImage | None = None
        self._photo_triple: ImageTk.PhotoImage | None = None
        self._preview_orig: Image.Image | None = None
        self._preview_edit: Image.Image | None = None

        self._build()
        self._set_status("Fotoğraf seçin (1–3 JPEG/PNG).")

    def _build(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Fotoğraf Seç…", command=self._pick_files).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(top, text="İşle / Önizle", command=self._process_active).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(top, text="Kaydet", command=self._save_active).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(top, text="Tümünü Kaydet", command=self._save_all).pack(
            side=tk.LEFT, padx=4
        )

        heif = "HEIC: hazır" if heif_available() else "HEIC: yok (pillow-heif)"
        ttk.Label(top, text=heif).pack(side=tk.RIGHT, padx=8)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(body, padding=4)
        right = ttk.Frame(body, padding=4)
        body.add(left, weight=1)
        body.add(right, weight=3)

        ttk.Label(left, text="Fotoğraflar", font=("", 11, "bold")).pack(anchor=tk.W)
        self.listbox = tk.Listbox(left, height=6, exportselection=False)
        self.listbox.pack(fill=tk.X, pady=4)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="Doğal Aydınlatma", font=("", 10, "bold")).pack(
            anchor=tk.W
        )

        self.mode_var = tk.StringVar(value=Mode.OTOMATIK.value)
        modes = [
            ("Otomatik", Mode.OTOMATIK.value),
            ("Hafif", Mode.HAFIF.value),
            ("Orta", Mode.ORTA.value),
            ("Güçlü", Mode.GUCLU.value),
        ]
        for label, val in modes:
            ttk.Radiobutton(
                left,
                text=label,
                value=val,
                variable=self.mode_var,
                command=self._sync_settings,
            ).pack(anchor=tk.W)

        ttk.Label(left, text="Aydınlatma şiddeti").pack(anchor=tk.W, pady=(6, 0))
        self.intensity_var = tk.DoubleVar(value=1.0)
        ttk.Scale(
            left,
            from_=0.4,
            to=1.4,
            variable=self.intensity_var,
            command=lambda _e: self._sync_settings(),
        ).pack(fill=tk.X)

        self.win_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            left,
            text="Pencere / parlak alan koruması",
            variable=self.win_var,
            command=self._sync_settings,
        ).pack(anchor=tk.W, pady=2)

        self.wb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left,
            text="Hafif beyaz dengesi (isteğe bağlı)",
            variable=self.wb_var,
            command=self._sync_settings,
        ).pack(anchor=tk.W)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(
            left, text="iPhone Tarzı – Natural HDR", font=("", 10, "bold")
        ).pack(anchor=tk.W)
        ttk.Label(
            left,
            text="(Look katmanı; Doğal Aydınlatma korunur)",
            font=("", 8),
        ).pack(anchor=tk.W)

        self.look_var = tk.StringVar(value=LookProfile.NONE.value)
        looks = [
            (LOOK_LABELS_TR[LookProfile.NONE], LookProfile.NONE.value),
            (LOOK_LABELS_TR[LookProfile.NATURAL], LookProfile.NATURAL.value),
            (
                LOOK_LABELS_TR[LookProfile.IPHONE_NATURAL_HDR],
                LookProfile.IPHONE_NATURAL_HDR.value,
            ),
            (
                LOOK_LABELS_TR[LookProfile.IPHONE_VIBRANT],
                LookProfile.IPHONE_VIBRANT.value,
            ),
        ]
        for label, val in looks:
            ttk.Radiobutton(
                left,
                text=label,
                value=val,
                variable=self.look_var,
                command=self._sync_settings,
            ).pack(anchor=tk.W)

        ttk.Label(left, text="Look gücü (0–100, varsayılan 50)").pack(
            anchor=tk.W, pady=(6, 0)
        )
        self.look_strength_var = tk.DoubleVar(value=50.0)
        ttk.Scale(
            left,
            from_=0.0,
            to=100.0,
            variable=self.look_strength_var,
            command=lambda _e: self._sync_settings(),
        ).pack(fill=tk.X)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        self.warn_text = tk.Text(left, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.warn_text.pack(fill=tk.BOTH, expand=True)

        # Preview area
        ttk.Label(right, text="Orijinal  |  Düzenlenmiş", font=("", 11, "bold")).pack(
            anchor=tk.W
        )
        prev = ttk.Frame(right)
        prev.pack(fill=tk.BOTH, expand=True)
        self.canvas_left = tk.Label(prev, background="#222", text="Orijinal")
        self.canvas_right = tk.Label(prev, background="#222", text="Düzenlenmiş")
        self.canvas_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        ttk.Label(
            right, text="Karşılaştırma kaydırıcısı (orijinal ← → düzenlenmiş)"
        ).pack(anchor=tk.W, pady=(6, 0))
        self.compare_var = tk.DoubleVar(value=0.5)
        ttk.Scale(
            right,
            from_=0.0,
            to=1.0,
            variable=self.compare_var,
            command=self._update_compare,
        ).pack(fill=tk.X)
        self.canvas_compare = tk.Label(right, background="#111", text="Karşılaştırma")
        self.canvas_compare.pack(fill=tk.X, pady=4)

        ttk.Label(
            right,
            text="Üçlü: Orijinal | Sadece Aydınlatma | iPhone Natural HDR",
            font=("", 10, "bold"),
        ).pack(anchor=tk.W, pady=(4, 0))
        self.canvas_triple = tk.Label(right, background="#0a0a0a", text="Üçlü önizleme")
        self.canvas_triple.pack(fill=tk.X, pady=4)

        self.status = ttk.Label(self, text="", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def _set_status(self, msg: str) -> None:
        self.status.config(text=msg)

    def _set_warnings(self, lines: list[str]) -> None:
        self.warn_text.config(state=tk.NORMAL)
        self.warn_text.delete("1.0", tk.END)
        self.warn_text.insert(tk.END, "\n".join(lines) if lines else "Uyarı yok.")
        self.warn_text.config(state=tk.DISABLED)

    def _pick_files(self) -> None:
        types = [
            ("Görüntüler", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"),
            ("Tümü", "*.*"),
        ]
        if heif_available():
            types.insert(0, ("HEIC", "*.heic *.heif *.HEIC"))
        paths = filedialog.askopenfilenames(title="Fotoğraf seç (1–3)", filetypes=types)
        if not paths:
            return
        if len(paths) > MAX_PHOTOS:
            messagebox.showwarning(
                "Limit", f"En fazla {MAX_PHOTOS} fotoğraf seçebilirsiniz."
            )
            paths = paths[:MAX_PHOTOS]
        self.slots = [PhotoSlot(Path(p)) for p in paths]
        self.listbox.delete(0, tk.END)
        for s in self.slots:
            self.listbox.insert(tk.END, s.path.name)
        self.listbox.selection_set(0)
        self.active_index = 0
        self._load_settings_to_ui()
        self._process_active()

    def _on_select(self, _evt: Any = None) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        self._sync_settings()
        self.active_index = int(sel[0])
        self._load_settings_to_ui()
        slot = self.slots[self.active_index]
        if slot.result is None:
            self._process_active()
        else:
            self._show_result(slot)

    def _load_settings_to_ui(self) -> None:
        if not self.slots:
            return
        s = self.slots[self.active_index].settings
        self.mode_var.set(s.mode.value)
        self.intensity_var.set(s.intensity)
        self.win_var.set(s.window_protection)
        self.wb_var.set(s.white_balance)
        self.look_var.set(s.look.value)
        self.look_strength_var.set(s.look_strength)

    def _sync_settings(self) -> None:
        if not self.slots:
            return
        slot = self.slots[self.active_index]
        slot.settings = EnhanceSettings(
            mode=Mode(self.mode_var.get()),
            intensity=float(self.intensity_var.get()),
            window_protection=bool(self.win_var.get()),
            white_balance=bool(self.wb_var.get()),
            look=LookProfile(self.look_var.get()),
            look_strength=float(self.look_strength_var.get()),
        )

    def _process_active(self) -> None:
        if not self.slots:
            messagebox.showinfo("Bilgi", "Önce fotoğraf seçin.")
            return
        self._sync_settings()
        slot = self.slots[self.active_index]
        self._set_status(f"İşleniyor: {slot.path.name}…")
        self.update_idletasks()
        try:
            result = process_file(slot.path, settings=slot.settings)
            slot.result = result
            self._show_result(slot)
            look_name = LOOK_LABELS_TR.get(slot.settings.look, slot.settings.look.value)
            band = (
                result.look_metrics.effect_band if result.look_metrics else "?"
            )
            self._set_status(
                f"Tamam: {slot.path.name} | "
                f"exp={result.applied['exposure']:.2f} "
                f"shadow={result.applied['shadow']:.2f} | look={look_name} "
                f"band={band}"
            )
        except Exception as exc:
            messagebox.showerror("Hata", str(exc))
            self._set_status("Hata oluştu.")

    def _show_result(self, slot: PhotoSlot) -> None:
        result = slot.result
        if result is None:
            return
        orig = _fit(_bgr_to_pil(result.original_bgr))
        edit = _fit(_bgr_to_pil(result.enhanced_bgr))
        if orig.size != edit.size:
            edit = edit.resize(orig.size, Image.Resampling.LANCZOS)
        self._preview_orig = orig
        self._preview_edit = edit
        self._photo_left = ImageTk.PhotoImage(orig)
        self._photo_right = ImageTk.PhotoImage(edit)
        self.canvas_left.config(image=self._photo_left, text="")
        self.canvas_right.config(image=self._photo_right, text="")
        self._update_compare()
        self._show_triple(slot)

        lines: list[str] = []
        a = result.analysis
        lines.append(
            f"Analiz: mean L={a.mean_luminance:.3f}, "
            f"gölge={a.shadow_ratio:.2f}, clip%={a.clipped_highlight_pct:.1f}, "
            f"gürültü={a.noise_risk:.2f}"
        )
        lines.append(
            f"Uygulanan: exposure={result.applied['exposure']:.2f} stop, "
            f"shadow={result.applied['shadow']:.2f}, local={result.applied['local']:.2f}"
        )
        lines.append(
            f"Look: {LOOK_LABELS_TR.get(result.settings.look, result.settings.look.value)} "
            f"(güç={result.settings.look_strength:.0f})"
        )
        if result.look_metrics is not None:
            lines.append("--- LOOK DEBUG ---")
            lines.extend(result.look_metrics.as_lines())
        q = result.quality
        if q.needs_manual_review:
            lines.append("--- UYARI / MANUEL KONTROL ---")
        for w in q.warnings:
            lines.append(f"• {w}")
        if not q.ok:
            lines.append("Kalite kapısı: kritik eşik aşıldı.")
        self._set_warnings(lines)

        if q.needs_manual_review:
            messagebox.showwarning(
                "Manuel kontrol önerilir",
                "\n".join(q.warnings[:4]) or "Kalite uyarıları var.",
            )

    def _show_triple(self, slot: PhotoSlot) -> None:
        result = slot.result
        if result is None or result.lit_bgr is None:
            return
        lit = result.lit_bgr
        hdr = apply_look(
            lit,
            LookProfile.IPHONE_NATURAL_HDR,
            slot.settings.look_strength,
            analysis=result.analysis,
            source_bgr=result.original_bgr,
        )
        order = (
            (result.original_bgr, "Orijinal"),
            (lit, "Sadece Aydınlatma"),
            (hdr, "iPhone Natural HDR"),
        )
        panels = [
            _label_strip(_fit(_bgr_to_pil(bgr), max_side=420), title)
            for bgr, title in order
        ]
        h = min(p.height for p in panels)
        panels = [
            p.crop((0, 0, p.width, h))
            if p.height == h
            else p.resize((int(p.width * h / p.height), h), Image.Resampling.LANCZOS)
            for p in panels
        ]
        gap = Image.new("RGB", (4, h), (40, 40, 40))
        triple = panels[0]
        for p in panels[1:]:
            row = Image.new("RGB", (triple.width + 4 + p.width, h), (20, 20, 20))
            row.paste(triple, (0, 0))
            row.paste(gap, (triple.width, 0))
            row.paste(p, (triple.width + 4, 0))
            triple = row
        self._photo_triple = ImageTk.PhotoImage(triple)
        self.canvas_triple.config(image=self._photo_triple, text="")

    def _update_compare(self, _evt: Any = None) -> None:
        if self._preview_orig is None or self._preview_edit is None:
            return
        t = float(self.compare_var.get())
        w, h = self._preview_orig.size
        split = int(w * t)
        blended = self._preview_orig.copy()
        right = self._preview_edit.crop((split, 0, w, h))
        blended.paste(right, (split, 0))
        if 0 < split < w:
            for y in range(h):
                blended.putpixel((split, y), (255, 220, 40))
        self._photo_blend = ImageTk.PhotoImage(blended)
        self.canvas_compare.config(image=self._photo_blend, text="")

    def _out_dir(self) -> Path:
        if self.slots:
            return default_output_dir(self.slots[0].path)
        return default_output_dir()

    def _save_active(self) -> None:
        if not self.slots:
            return
        slot = self.slots[self.active_index]
        if slot.result is None:
            self._process_active()
        if slot.result is None:
            return
        out = save_result(slot.result, self._out_dir(), source_path=slot.path)
        messagebox.showinfo("Kaydedildi", f"Kayıt:\n{out}")
        self._set_status(f"Kaydedildi: {out}")

    def _save_all(self) -> None:
        if not self.slots:
            return
        out_dir = self._out_dir()
        saved = []
        for i, slot in enumerate(self.slots):
            self.active_index = i
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(i)
            self._load_settings_to_ui()
            if slot.result is None:
                self._process_active()
            if slot.result is None:
                continue
            saved.append(save_result(slot.result, out_dir, source_path=slot.path))
        messagebox.showinfo(
            "Kaydedildi",
            "Kayıtlar:\n" + "\n".join(str(p) for p in saved) if saved else "Kayıt yok",
        )


def run_app() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    run_app()
