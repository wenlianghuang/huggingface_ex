from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

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

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    prompt = data['prompt']
    
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.5,
        "do_sample": False,
    }

    output = pipe(prompt, **generation_args)
    generated_text = output[0]['generated_text']
    
    return jsonify({"response": generated_text})

if __name__ == '__main__':
    app.run(debug=True)
