import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

def generate_text_stream(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    model_output = model.generate(input_ids, max_length=256, do_sample=False)
    decoded_output = tokenizer.decode(model_output[0], skip_special_tokens=True)

    for char in decoded_output:
        yield char
        time.sleep(0.05)  # Adjust speed here

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user_submit(user_message, history):
        history.append([user_message, ""])
        return "", history

    def bot_response(history):
        user_message = history[-1][0]
        bot_message = generate_text_stream(user_message)
        for char in bot_message:
            history[-1][1] += char
            yield history

    def clear_click():
        return [], []

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
        bot_response, chatbot, chatbot
    )
    clear.click(clear_click, None, chatbot)

demo.queue()
demo.launch()
