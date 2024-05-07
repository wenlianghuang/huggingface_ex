import torch 
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline

model = "C:\\Users\\acer alan\Desktop\\Phi-3-mini-128k"

model = AutoModelForCausalLM.from_pretrained(
    "C:\\Users\\acer alan\Desktop\\Phi-3-mini-128k",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
#tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\acer alan\Desktop\\Phi-3-mini-128k")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

''''
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
    #print("Chatbot:", sequences[1]['generated_text'])
    print(sequences[0]['generated_text'])
'''
messages = [
    {"role":"system","content":"You are a reporter"},
    {"role":"user","content":"The capital of China is?"},
]
generation_args = {
    "max_new_tokens": 256,
    "return_full_text": False,
    "temperature": 0.5,
    "do_sample": False,
    
}
while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye','quit','exit']:
        print("Chatbot: Goodbye")
        break
    messages[1]['content'] = user_input
    #print(message)
    
    output = pipe(messages,**generation_args)
    #output = pipe(str(user_input),**generation_args)
    #print(type(output[0]['generated_text']))
    print(output[0]['generated_text'])
    