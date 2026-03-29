"""
DocMind_AI - RAG-based Document Q&A Application
Main application entry point with GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
from pathlib import Path

from rag_engine import RAGEngine


class DocMindApp:
    """Main GUI Application for DocMind_AI"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DocMind AI — Document Intelligence")
        self.root.geometry("1000x750")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)

        self.rag_engine = RAGEngine()
        self.loaded_file = None
        self.chat_history = []

        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._apply_styles()
        self._build_header()
        self._build_main_area()
        self._build_status_bar()

    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Card.TFrame",    background="#16213e", relief="flat")
        style.configure("Main.TFrame",    background="#1a1a2e")
        style.configure("Header.TFrame",  background="#0f3460")

        style.configure("Title.TLabel",
                        background="#0f3460", foreground="#e94560",
                        font=("Segoe UI", 18, "bold"))
        style.configure("Sub.TLabel",
                        background="#0f3460", foreground="#a8b2d8",
                        font=("Segoe UI", 10))
        style.configure("Section.TLabel",
                        background="#16213e", foreground="#ccd6f6",
                        font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel",
                        background="#0d0d1a", foreground="#64ffda",
                        font=("Segoe UI", 9))
        style.configure("Info.TLabel",
                        background="#16213e", foreground="#8892b0",
                        font=("Segoe UI", 9))

        style.configure("Upload.TButton",
                        background="#e94560", foreground="white",
                        font=("Segoe UI", 10, "bold"),
                        padding=(12, 8), relief="flat")
        style.map("Upload.TButton",
                  background=[("active", "#c73652")])

        style.configure("Send.TButton",
                        background="#64ffda", foreground="#0d0d1a",
                        font=("Segoe UI", 10, "bold"),
                        padding=(12, 8), relief="flat")
        style.map("Send.TButton",
                  background=[("active", "#4cd9b8")])

        style.configure("Clear.TButton",
                        background="#303050", foreground="#ccd6f6",
                        font=("Segoe UI", 9),
                        padding=(8, 6), relief="flat")
        style.map("Clear.TButton",
                  background=[("active", "#404060")])

    def _build_header(self):
        hdr = ttk.Frame(self.root, style="Header.TFrame")
        hdr.pack(fill="x", padx=0, pady=0)

        ttk.Label(hdr, text="⚡ DocMind AI", style="Title.TLabel").pack(
            side="left", padx=20, pady=15)
        ttk.Label(hdr, text="RAG-Powered Document Intelligence",
                  style="Sub.TLabel").pack(side="left", padx=5, pady=15)

        self.model_var = tk.StringVar(value="minimax-m2.7:cloud")
        model_frame = ttk.Frame(hdr, style="Header.TFrame")
        model_frame.pack(side="right", padx=20, pady=10)
        tk.Label(model_frame, text="Chat Model:", bg="#0f3460",
                 fg="#a8b2d8", font=("Segoe UI", 9)).pack(side="left")
        model_cb = ttk.Combobox(model_frame, textvariable=self.model_var,
                                values=["minimax-m2.7:cloud"],
                                width=18, state="readonly")
        model_cb.pack(side="left", padx=(5, 0))
        model_cb.bind("<<ComboboxSelected>>", self._on_model_change)

    def _build_main_area(self):
        main = ttk.Frame(self.root, style="Main.TFrame")
        main.pack(fill="both", expand=True, padx=15, pady=10)
        main.columnconfigure(0, weight=1, minsize=280)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        self._build_left_panel(main)
        self._build_right_panel(main)

    def _build_left_panel(self, parent):
        left = ttk.Frame(parent, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        left.columnconfigure(0, weight=1)

        # ── Document Upload ──
        ttk.Label(left, text="📄  Document", style="Section.TLabel").pack(
            anchor="w", padx=15, pady=(15, 8))

        upload_btn = ttk.Button(left, text="＋  Upload Document",
                                style="Upload.TButton",
                                command=self._upload_document)
        upload_btn.pack(fill="x", padx=15, pady=(0, 8))

        self.file_label = tk.Label(left, text="No file loaded",
                                   bg="#16213e", fg="#8892b0",
                                   font=("Segoe UI", 9), wraplength=220,
                                   justify="left")
        self.file_label.pack(anchor="w", padx=15)

        # ── Separator ──
        tk.Frame(left, bg="#2a2a4a", height=1).pack(fill="x", padx=15, pady=12)

        # ── Chunking Settings ──
        ttk.Label(left, text="⚙️  Chunking Settings",
                  style="Section.TLabel").pack(anchor="w", padx=15, pady=(0, 8))

        self._labeled_scale(left, "Chunk Size", 200, 2000, 800, "chunk_size")
        self._labeled_scale(left, "Chunk Overlap", 0, 400, 150, "chunk_overlap")
        self._labeled_scale(left, "Top-K Results", 1, 10, 4, "top_k")

        # ── Process Button ──
        tk.Frame(left, bg="#2a2a4a", height=1).pack(fill="x", padx=15, pady=12)

        self.process_btn = ttk.Button(left, text="⚡  Process Document",
                                      style="Upload.TButton",
                                      command=self._process_document,
                                      state="disabled")
        self.process_btn.pack(fill="x", padx=15, pady=(0, 8))

        # ── Progress ──
        self.progress = ttk.Progressbar(left, mode="indeterminate",
                                        style="TProgressbar")
        self.progress.pack(fill="x", padx=15, pady=(0, 8))

        # ── Doc Info ──
        tk.Frame(left, bg="#2a2a4a", height=1).pack(fill="x", padx=15, pady=4)
        ttk.Label(left, text="📊  Document Info",
                  style="Section.TLabel").pack(anchor="w", padx=15, pady=(8, 4))

        self.doc_info = tk.Text(left, height=6, bg="#0d0d1a", fg="#64ffda",
                                font=("Consolas", 8), relief="flat",
                                state="disabled", wrap="word")
        self.doc_info.pack(fill="x", padx=15, pady=(0, 15))

    def _labeled_scale(self, parent, label, from_, to, default, attr):
        frame = tk.Frame(parent, bg="#16213e")
        frame.pack(fill="x", padx=15, pady=2)

        var = tk.IntVar(value=default)
        setattr(self, f"{attr}_var", var)

        tk.Label(frame, text=label, bg="#16213e", fg="#a8b2d8",
                 font=("Segoe UI", 8)).pack(anchor="w")

        row = tk.Frame(frame, bg="#16213e")
        row.pack(fill="x")

        scale = tk.Scale(row, from_=from_, to=to, orient="horizontal",
                         variable=var, bg="#16213e", fg="#ccd6f6",
                         troughcolor="#2a2a4a", highlightthickness=0,
                         relief="flat", showvalue=False, length=160)
        scale.pack(side="left", fill="x", expand=True)

        val_lbl = tk.Label(row, textvariable=var, bg="#16213e", fg="#64ffda",
                           font=("Consolas", 9), width=5)
        val_lbl.pack(side="right")

    def _build_right_panel(self, parent):
        right = ttk.Frame(parent, style="Card.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ── Chat header ──
        chat_hdr = tk.Frame(right, bg="#16213e")
        chat_hdr.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 8))

        tk.Label(chat_hdr, text="💬  Chat", bg="#16213e", fg="#ccd6f6",
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Button(chat_hdr, text="Clear Chat", style="Clear.TButton",
                   command=self._clear_chat).pack(side="right")

        # ── Chat Display ──
        chat_frame = tk.Frame(right, bg="#16213e")
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=15)
        chat_frame.rowconfigure(0, weight=1)
        chat_frame.columnconfigure(0, weight=1)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap="word", bg="#0d0d1a", fg="#ccd6f6",
            font=("Segoe UI", 10), relief="flat", state="disabled",
            insertbackground="white", selectbackground="#233554",
            padx=12, pady=10)
        self.chat_display.grid(row=0, column=0, sticky="nsew")

        # Configure chat tags
        self.chat_display.tag_configure("user_tag",
                                        foreground="#64ffda",
                                        font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_configure("ai_tag",
                                        foreground="#e94560",
                                        font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_configure("user_msg",
                                        foreground="#ccd6f6",
                                        font=("Segoe UI", 10))
        self.chat_display.tag_configure("ai_msg",
                                        foreground="#a8b2d8",
                                        font=("Segoe UI", 10))
        self.chat_display.tag_configure("source_tag",
                                        foreground="#8892b0",
                                        font=("Segoe UI", 8, "italic"))
        self.chat_display.tag_configure("divider",
                                        foreground="#2a2a4a",
                                        font=("Segoe UI", 6))
        self.chat_display.tag_configure("error_tag",
                                        foreground="#ff6b6b",
                                        font=("Segoe UI", 10))

        # ── Input Area ──
        input_frame = tk.Frame(right, bg="#16213e")
        input_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=12)
        input_frame.columnconfigure(0, weight=1)

        self.query_entry = tk.Text(input_frame, height=3, bg="#0d0d1a",
                                   fg="#ccd6f6", font=("Segoe UI", 10),
                                   relief="flat", insertbackground="white",
                                   wrap="word", padx=10, pady=8)
        self.query_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.query_entry.bind("<Return>", self._on_enter)
        self.query_entry.bind("<Shift-Return>", lambda e: None)

        tk.Label(input_frame, text="Enter ↵ to send  •  Shift+Enter for newline",
                 bg="#16213e", fg="#4a5568",
                 font=("Segoe UI", 8)).grid(row=1, column=0, sticky="w", pady=(3, 0))

        btn_col = tk.Frame(input_frame, bg="#16213e")
        btn_col.grid(row=0, column=1, sticky="n")
        ttk.Button(btn_col, text="Send ➤", style="Send.TButton",
                   command=self._send_query).pack(pady=(0, 4))
        ttk.Button(btn_col, text="Clear", style="Clear.TButton",
                   command=lambda: self.query_entry.delete("1.0", "end")).pack()

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg="#0d0d1a", height=28)
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready  •  Upload a document to begin")
        tk.Label(bar, textvariable=self.status_var, bg="#0d0d1a",
                 fg="#64ffda", font=("Segoe UI", 9),
                 anchor="w").pack(side="left", padx=15, pady=4)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_model_change(self, _event=None):
        model = self.model_var.get()
        self.rag_engine.set_model(model)
        self._set_status(f"Model switched to: {model}")

    def _upload_document(self):
        path = filedialog.askopenfilename(
            title="Select a Document",
            filetypes=[
                ("Supported files", "*.pdf *.txt *.csv *.docx *.md"),
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("Word Documents", "*.docx"),
                ("Markdown", "*.md"),
                ("All files", "*.*"),
            ])
        if path:
            self.loaded_file = path
            fname = Path(path).name
            size_kb = os.path.getsize(path) / 1024
            self.file_label.config(
                text=f"📄 {fname}\n({size_kb:.1f} KB)", fg="#64ffda")
            self.process_btn.config(state="normal")
            self._set_status(f"File loaded: {fname}")

    def _process_document(self):
        if not self.loaded_file:
            messagebox.showwarning("No File", "Please upload a document first.")
            return

        self.process_btn.config(state="disabled")
        self.progress.start(10)
        self._set_status("Processing document… please wait")

        def worker():
            try:
                info = self.rag_engine.load_and_index(
                    file_path=self.loaded_file,
                    chunk_size=self.chunk_size_var.get(),
                    chunk_overlap=self.chunk_overlap_var.get(),
                    model_name=self.model_var.get(),
                )
                self.root.after(0, self._on_process_done, info)
            except Exception as exc:
                self.root.after(0, self._on_process_error, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_process_done(self, info: dict):
        self.progress.stop()
        self.process_btn.config(state="normal")
        self._set_status("✅ Document indexed — ready to answer questions!")

        self.doc_info.config(state="normal")
        self.doc_info.delete("1.0", "end")
        self.doc_info.insert("end",
            f"Pages/rows  : {info.get('pages', 'N/A')}\n"
            f"Chunks      : {info.get('chunks', 'N/A')}\n"
            f"Chunk size  : {info.get('chunk_size')}\n"
            f"Overlap     : {info.get('chunk_overlap')}\n"
            f"Loader      : {info.get('loader')}\n"
            f"Chat model  : {info.get('chat_model')}\n"
            f"Embed model : {info.get('embed_model')}\n"
        )
        self.doc_info.config(state="disabled")

        self._append_chat("system",
            f"✅ Document '{Path(self.loaded_file).name}' has been processed "
            f"({info.get('chunks')} chunks). You can now ask questions!", "ai_tag", "ai_msg")

    def _on_process_error(self, error: str):
        self.progress.stop()
        self.process_btn.config(state="normal")
        self._set_status(f"❌ Error: {error}")
        messagebox.showerror("Processing Error", error)

    def _on_enter(self, event):
        if not event.state & 0x1:          # Shift not held
            self._send_query()
            return "break"

    def _send_query(self):
        query = self.query_entry.get("1.0", "end").strip()
        if not query:
            return
        if not self.rag_engine.is_ready():
            messagebox.showinfo("Not Ready",
                                "Please process a document first.")
            return

        self.query_entry.delete("1.0", "end")
        self._append_chat("You", query, "user_tag", "user_msg")
        self._set_status("Thinking…")

        def worker():
            try:
                result = self.rag_engine.query(
                    query, top_k=self.top_k_var.get())
                self.root.after(0, self._on_answer, result)
            except Exception as exc:
                self.root.after(0, self._on_query_error, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_answer(self, result: dict):
        answer   = result.get("answer", "No answer returned.")
        sources  = result.get("sources", [])

        self._append_chat("DocMind", answer, "ai_tag", "ai_msg")

        if sources:
            src_text = "\n📎 Sources: " + " | ".join(
                f"[{i+1}] {s}" for i, s in enumerate(sources[:3]))
            self._append_chat("", src_text, "", "source_tag")

        self._set_status("✅ Answer ready")

    def _on_query_error(self, error: str):
        self._append_chat("Error", error, "ai_tag", "error_tag")
        self._set_status(f"❌ Query error: {error}")

    def _append_chat(self, speaker: str, message: str,
                     speaker_tag: str, msg_tag: str):
        self.chat_display.config(state="normal")
        if speaker:
            self.chat_display.insert("end", f"\n{speaker}:  ", speaker_tag)
        self.chat_display.insert("end", f"{message}\n", msg_tag)
        self.chat_display.insert("end", "─" * 60 + "\n", "divider")
        self.chat_display.see("end")
        self.chat_display.config(state="disabled")

    def _clear_chat(self):
        self.chat_display.config(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.config(state="disabled")

    def _set_status(self, msg: str):
        self.status_var.set(msg)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = DocMindApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()