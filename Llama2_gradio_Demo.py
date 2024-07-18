import gradio as gr
from transformers import AutoTokenizer
import torch
from transformers import pipeline
import time

model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def get_llama_response(prompt: str) -> str:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        str: The model's response.
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=False,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256,
    )
    return sequences[0]['generated_text']

# 使用Gradio的Blocks API来创建界面
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user_submit(user_message, history):
        history.append([user_message, None])  # Append user message
        return "", history

    def bot_response(history):
        user_message = history[-1][0]
        bot_message = get_llama_response(user_message)
        history[-1][1] = bot_message  # Update with bot message
        return history

    def clear_click():
        return [], []

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
        bot_response, chatbot, chatbot
    )
    clear.click(clear_click, None, chatbot)

# 启动Gradio界面
demo.queue()
demo.launch()
