"""
DocuMind AI - Main GUI Application

A document reader with RAG-based Q&A, quiz generation, and calculator agent.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from typing import Optional, Tuple
import threading

from app.models import test_ollama_connection
from app.document_processor import DocumentProcessor
from app.database import get_database
from app.chains import get_qa_chain, get_quiz_chain, reset_chains


class DocuMindGUI:
    """Main GUI application for DocuMind AI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DocuMind AI - Document Reader with RAG")
        self.root.geometry("900x800")
        self.root.minsize(700, 600)

        # Initialize components
        self.db = get_database()
        self.qa_chain = get_qa_chain()
        self.quiz_chain = get_quiz_chain()

        # State
        self.current_doc_id: Optional[int] = None
        self.current_doc_name: Optional[str] = None
        self.quiz_mode = False
        self.quiz_questions: list = []
        self.user_answers: dict = {}

        # Setup GUI
        self._setup_styles()
        self._create_widgets()
        self._refresh_document_list()

        # Check Ollama connection in background
        self._check_ollama_async()

    def _setup_styles(self):
        """Configure ttk styles for the application."""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure colors
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"))
        style.configure("Title.TLabel", font=("Helvetica", 18, "bold"))
        style.configure("Doc.TFrame", background="#f0f0f0")

    def _create_widgets(self):
        """Create and layout all GUI widgets."""
        # Main container with padding
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # ===== HEADER =====
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(
            header_frame,
            text="DocuMind AI",
            style="Title.TLabel"
        )
        title_label.pack(side=tk.LEFT)

        clear_btn = ttk.Button(
            header_frame,
            text="Clear Chat",
            command=self._clear_chat
        )
        clear_btn.pack(side=tk.RIGHT)

        # ===== DOCUMENT PANEL =====
        doc_frame = ttk.LabelFrame(
            main_container,
            text="Document Management",
            padding="10"
        )
        doc_frame.pack(fill=tk.X, pady=(0, 10))

        # Document controls row
        doc_controls = ttk.Frame(doc_frame)
        doc_controls.pack(fill=tk.X)

        self.load_btn = ttk.Button(
            doc_controls,
            text="Load Document",
            command=self._load_document
        )
        self.load_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.current_doc_label = ttk.Label(
            doc_controls,
            text="No document loaded",
            foreground="gray"
        )
        self.current_doc_label.pack(side=tk.LEFT, padx=(0, 20))

        self.doc_count_label = ttk.Label(
            doc_controls,
            text="Documents in DB: 0"
        )
        self.doc_count_label.pack(side=tk.LEFT, padx=(0, 20))

        self.clear_db_btn = ttk.Button(
            doc_controls,
            text="Clear Database",
            command=self._clear_database
        )
        self.clear_db_btn.pack(side=tk.LEFT)

        # Document list
        list_frame = ttk.Frame(doc_frame)
        list_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(list_frame, text="Loaded Documents:").pack(anchor=tk.W)

        self.doc_listbox = tk.Listbox(list_frame, height=4, selectmode=tk.SINGLE)
        self.doc_listbox.pack(fill=tk.X, pady=(5, 0))
        self.doc_listbox.bind("<<ListboxSelect>>", self._on_doc_select)

        # ===== CHAT AREA =====
        chat_frame = ttk.LabelFrame(
            main_container,
            text="Chat / Quiz Area",
            padding="10"
        )
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.chat_text = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            state=tk.DISABLED,
            bg="#ffffff"
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for styling
        self.chat_text.tag_configure("user", foreground="#0066cc", font=("Helvetica", 11, "bold"))
        self.chat_text.tag_configure("bot", foreground="#333333")
        self.chat_text.tag_configure("system", foreground="#666666", font=("Helvetica", 10, "italic"))
        self.chat_text.tag_configure("quiz_question", foreground="#660066")
        self.chat_text.tag_configure("quiz_answer", foreground="#006600")
        self.chat_text.tag_configure("source", foreground="#888888", font=("Helvetica", 9))

        # ===== QUIZ PANEL =====
        quiz_frame = ttk.LabelFrame(
            main_container,
            text="Quiz Mode",
            padding="10"
        )
        quiz_frame.pack(fill=tk.X, pady=(0, 10))

        quiz_controls = ttk.Frame(quiz_frame)
        quiz_controls.pack(fill=tk.X)

        # Quiz type selection
        ttk.Label(quiz_controls, text="Quiz Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.quiz_type_var = tk.StringVar(value="Multiple Choice")
        quiz_type_combo = ttk.Combobox(
            quiz_controls,
            textvariable=self.quiz_type_var,
            values=["Multiple Choice", "True/False", "Short Answer"],
            state="readonly",
            width=15
        )
        quiz_type_combo.pack(side=tk.LEFT, padx=(0, 20))

        # Number of questions
        ttk.Label(quiz_controls, text="Questions:").pack(side=tk.LEFT, padx=(0, 5))
        self.num_questions_var = tk.IntVar(value=5)
        questions_spin = ttk.Spinbox(
            quiz_controls,
            from_=3,
            to=10,
            textvariable=self.num_questions_var,
            width=5
        )
        questions_spin.pack(side=tk.LEFT, padx=(0, 20))

        # Quiz buttons
        self.generate_quiz_btn = ttk.Button(
            quiz_controls,
            text="Generate Quiz",
            command=self._generate_quiz
        )
        self.generate_quiz_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.submit_quiz_btn = ttk.Button(
            quiz_controls,
            text="Submit Quiz",
            command=self._submit_quiz,
            state=tk.DISABLED
        )
        self.submit_quiz_btn.pack(side=tk.LEFT)

        # Status label for quiz
        self.quiz_status_label = ttk.Label(
            quiz_controls,
            text="",
            foreground="blue"
        )
        self.quiz_status_label.pack(side=tk.LEFT, padx=(20, 0))

        # ===== INPUT AREA =====
        input_frame = ttk.Frame(main_container)
        input_frame.pack(fill=tk.X)

        self.input_entry = ttk.Entry(
            input_frame,
            font=("Helvetica", 12)
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_entry.bind("<Return>", self._on_input_submit)

        self.send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self._on_input_submit
        )
        self.send_btn.pack(side=tk.LEFT)

    def _check_ollama_async(self):
        """Check Ollama connection in background thread."""
        def check():
            if test_ollama_connection():
                self.root.after(0, lambda: self._add_system_message("Ollama connected successfully."))
            else:
                self.root.after(0, lambda: self._add_system_message(
                    "Warning: Could not connect to Ollama. Please ensure Ollama is running."
                ))
        threading.Thread(target=check, daemon=True).start()

    def _add_message(self, message: str, tag: str = "bot"):
        """Add a message to the chat area."""
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.insert(tk.END, message + "\n\n", tag)
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def _add_system_message(self, message: str):
        """Add a system message to the chat area."""
        self._add_message(f"[System] {message}", "system")

    def _clear_chat(self):
        """Clear the chat area and reset QA chain history."""
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.delete(1.0, tk.END)
        self.chat_text.configure(state=tk.DISABLED)
        self.qa_chain.clear_history()
        self._add_system_message("Chat history cleared.")

    def _refresh_document_list(self):
        """Refresh the document listbox."""
        self.doc_listbox.delete(0, tk.END)
        docs = self.db.get_all_documents()

        for doc in docs:
            self.doc_listbox.insert(tk.END, f"{doc[1]} (ID: {doc[0]}, Chunks: {doc[4]})")

        self.doc_count_label.config(text=f"Documents in DB: {len(docs)}")

    def _load_document(self):
        """Load a document from file."""
        filepath = filedialog.askopenfilename(
            title="Select a document",
            filetypes=[
                ("Document files", "*.pdf *.docx *.txt"),
                ("PDF files", "*.pdf"),
                ("Word files", "*.docx"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if not filepath:
            return

        # Process document in background
        self._add_system_message(f"Loading document: {filepath.split('/')[-1]}...")
        self.load_btn.config(state=tk.DISABLED)

        def process():
            try:
                # Process document
                chunks = DocumentProcessor.process_document(filepath)

                if not chunks:
                    self.root.after(0, lambda: self._add_system_message("No content found in document."))
                    self.root.after(0, lambda: self.load_btn.config(state=tk.NORMAL))
                    return

                # Store in database
                filename = filepath.split('/')[-1]
                doc_id = self.db.add_document(filename, filepath, chunks)

                # Update UI
                def update_ui():
                    self.current_doc_id = doc_id
                    self.current_doc_name = filename
                    self.current_doc_label.config(
                        text=f"Current: {filename}",
                        foreground="black"
                    )
                    self._refresh_document_list()
                    self._add_system_message(
                        f"Document '{filename}' loaded successfully. "
                        f"Created {len(chunks)} chunks."
                    )
                    self.load_btn.config(state=tk.NORMAL)

                self.root.after(0, update_ui)

            except Exception as e:
                def update_ui():
                    self._add_system_message(f"Error loading document: {str(e)}")
                    self.load_btn.config(state=tk.NORMAL)
                self.root.after(0, update_ui)

        threading.Thread(target=process, daemon=True).start()

    def _on_doc_select(self, event):
        """Handle document selection from listbox."""
        selection = self.doc_listbox.curselection()
        if selection:
            doc_info = self.doc_listbox.get(selection[0])
            # Extract doc_id from the string
            import re
            match = re.search(r'ID:\s*(\d+)', doc_info)
            if match:
                self.current_doc_id = int(match.group(1))
                self.current_doc_name = doc_info.split(" (")[0]
                self.current_doc_label.config(text=f"Current: {self.current_doc_name}")

    def _clear_database(self):
        """Clear all data from the database."""
        if messagebox.askyesno("Confirm", "Are you sure you want to delete all documents?"):
            self.db.clear_all_data()
            reset_chains()
            self.qa_chain = get_qa_chain()
            self.quiz_chain = get_quiz_chain()
            self.current_doc_id = None
            self.current_doc_name = None
            self.current_doc_label.config(text="No document loaded", foreground="gray")
            self._refresh_document_list()
            self._clear_chat()
            self._add_system_message("Database cleared.")

    def _generate_quiz(self):
        """Generate a quiz from the current document."""
        if self.current_doc_id is None:
            # If no document selected, try to use the first one
            docs = self.db.get_all_documents()
            if not docs:
                self._add_system_message("Please load a document first.")
                return
            self.current_doc_id = docs[0][0]
            self.current_doc_name = docs[0][1]

        self._add_system_message("Generating quiz...")
        self.generate_quiz_btn.config(state=tk.DISABLED)
        self.submit_quiz_btn.config(state=tk.DISABLED)

        def generate():
            try:
                quiz_type = self.quiz_type_var.get()
                num_questions = self.num_questions_var.get()

                quiz_text = self.quiz_chain.generate_quiz(
                    doc_id=self.current_doc_id,
                    quiz_type=quiz_type,
                    num_questions=num_questions
                )

                def update_ui():
                    self._add_message("=== GENERATED QUIZ ===", "system")
                    self._add_message(quiz_text, "quiz_question")
                    self.quiz_mode = True
                    self.submit_quiz_btn.config(state=tk.NORMAL)
                    self.generate_quiz_btn.config(state=tk.NORMAL)
                    self.quiz_status_label.config(text="Quiz ready - enter answers below")

                self.root.after(0, update_ui)

            except Exception as e:
                def update_ui():
                    self._add_system_message(f"Error generating quiz: {str(e)}")
                    self.generate_quiz_btn.config(state=tk.NORMAL)
                self.root.after(0, update_ui)

        threading.Thread(target=generate, daemon=True).start()

    def _submit_quiz(self):
        """Submit quiz answers for grading."""
        # Get answers from input and parse them
        # Format expected: "1:A, 2:B, 3:C" or similar
        input_text = self.input_entry.get().strip()
        self.input_entry.delete(0, tk.END)

        if not input_text:
            self._add_system_message("Please enter your quiz answers first.")
            return

        # Parse the answers
        import re
        answer_pattern = r'(\d+)\s*:\s*([A-Da-d])'
        matches = re.findall(answer_pattern, input_text)

        if not matches:
            self._add_system_message(
                "Please enter answers in format: 1:A, 2:B, 3:C (question number : answer letter)"
            )
            return

        user_answers = {int(q): a.upper() for q, a in matches}

        # Grade the quiz
        score, feedback, correct = self.quiz_chain.check_answers(user_answers)

        # Save result to database
        self.db.save_quiz_result(self.current_doc_id, score, len(correct))

        # Display results
        self._add_message("\n=== QUIZ SUBMISSION ===", "system")
        self._add_message(feedback, "quiz_answer")
        self._add_system_message("Quiz results saved.")

        # Reset quiz mode
        self.quiz_mode = False
        self.submit_quiz_btn.config(state=tk.DISABLED)
        self.quiz_status_label.config(text="")

    def _on_input_submit(self, event=None):
        """Handle input submission."""
        user_input = self.input_entry.get().strip()
        if not user_input:
            return

        self.input_entry.delete(0, tk.END)

        # Display user message
        self._add_message(f"You: {user_input}", "user")

        # Check if we need to process a quiz answer
        if self.quiz_mode:
            self.input_entry.insert(0, user_input)
            self._submit_quiz()
            return

        # Process question in background
        def ask():
            try:
                answer, sources = self.qa_chain.ask_question(user_input)

                def update_ui():
                    self._add_message(f"Bot: {answer}", "bot")

                    # Show sources if available
                    if sources:
                        self._add_message("\n--- Sources ---", "system")
                        for i, source in enumerate(sources[:3], 1):
                            self._add_message(f"{i}. {source}", "source")

                self.root.after(0, update_ui)

            except Exception as e:
                def update_ui():
                    self._add_system_message(f"Error: {str(e)}")
                self.root.after(0, update_ui)

        threading.Thread(target=ask, daemon=True).start()


def main():
    """Main entry point."""
    root = tk.Tk()

    # Set icon (optional, will work without it)
    try:
        root.iconbitmap("icon.ico")
    except:
        pass

    app = DocuMindGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
