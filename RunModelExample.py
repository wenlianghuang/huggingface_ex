import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 設置模型和 tokenizer 的路徑
model_path = "./sample_phi3_finetune_example"  # 你可以替換為你的模型的實際路徑或 Hugging Face Hub 的模型名稱

# 載入模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 設置模型為評估模式
model.eval()

def generate_answer(question, model, tokenizer, max_length=400, num_return_sequences=1):
    # 編碼輸入問題
    inputs = tokenizer.encode(question, return_tensors='pt')
    
    # 生成答案
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences, do_sample=False)
    
    # 解碼生成的答案
    answers = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return answers

# 示例問題
#question = "How has the new training facility impacted the team's physical performance?"
question = "Which type of rock is commonly used for construction and why?"
# 生成答案
answers = generate_answer(question, model, tokenizer)

# 輸出答案
for idx, answer in enumerate(answers):
    print(f"Answer {idx+1}: {answer}")