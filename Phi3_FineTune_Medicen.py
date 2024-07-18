import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from transformers import BitsAndBytesConfig

#model = "C:\\Users\\acer alan\Desktop\\Phi-3-mini-128k"
#model_id = "wenlianghuang/phi-3-matt-medicine-election"
model_id = "wenlianghuang/phi-3-matt-medicine"
#model_id = "wenlianghuang/phi-3-Taiwan-election"
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_compute_dtype='float16',
                                #bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
#tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
#tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\acer alan\Desktop\\Phi-3-mini-128k",add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "right"
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

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
    print("Chatbot: ",output[0]['generated_text'])