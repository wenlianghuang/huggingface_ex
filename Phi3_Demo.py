import torch 
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline


model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def get_Phi_Example(prompt: str) -> None:
    """
    Generate a response from the phi3 model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    sequences = pipe(
        prompt,
        do_sample=False,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256,
        temperature=0.5,
    )
    print("Chatbot:", sequences[0]['generated_text'])

while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye','quit','exit']:
        print("Chatbot: Goodbye")
        break
    get_Phi_Example(user_input)
    