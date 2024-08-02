import sys
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTextEdit, QPushButton, QWidget, QLineEdit
from PyQt5.QtCore import QThread, pyqtSignal
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from threading import Event

class TextStreamerToSignal:
    def __init__(self, output_signal):
        self.output_signal = output_signal

    def __call__(self, text):
        self.output_signal.emit(text)

class LLMThread(QThread):
    output_signal = pyqtSignal(str)

    def __init__(self, model, tokenizer, prompt):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def run(self):
        self.stop_event = Event()
        streamer = TextStreamerToSignal(self.output_signal)
        messages = [
            {"role": "system", "content": "You are a reporter"},
            {"role": "user", "content": self.prompt},
        ]
        generation_args = {
            "max_new_tokens": 256,
            "return_full_text": False,
            "temperature": 0.5,
            "do_sample": False,
        }
        output = self.pipe(messages, **generation_args)
        for token in output[0]['generated_text']:
            if self.stop_event.is_set():
                break
            streamer(token)

    def stop(self):
        self.stop_event.set()

class MainWindow(QMainWindow):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.initUI()

    def initUI(self):
        self.setWindowTitle("LLM Streamer")
        self.setGeometry(100, 100, 800, 600)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("font-size: 16px;")

        self.input_line = QLineEdit(self)
        self.input_line.setPlaceholderText("Enter your question here...")

        self.button = QPushButton("Generate", self)
        self.button.clicked.connect(self.start_streaming)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.input_line)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def start_streaming(self):
        prompt = self.input_line.text()
        if not prompt:
            return
        self.text_edit.clear()
        self.llm_thread = LLMThread(self.model, self.tokenizer, prompt)
        self.llm_thread.output_signal.connect(self.update_text_edit)
        self.llm_thread.start()

    def update_text_edit(self, text):
        self.text_edit.insertPlainText(text)
        self.text_edit.verticalScrollBar().setValue(self.text_edit.verticalScrollBar().maximum())

if __name__ == "__main__":
    model_path = "E:\\Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    app = QApplication(sys.argv)
    mainWindow = MainWindow(model, tokenizer)
    mainWindow.show()
    sys.exit(app.exec_())
