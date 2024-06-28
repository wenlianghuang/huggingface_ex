import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 設置模型和 tokenizer 的名稱
model_name = "wenlianghuang/sample_phi3_finetune_example"  # 替換為你在 Hugging Face Hub 上的模型名稱

# 設置模型參數，包括 flash attention 的支援
model_kwargs = {
    "use_cache": False,
    "trust_remote_code": True,
    "attn_implementation": "flash_attention_2",  # 使用 flash attention
    "torch_dtype": torch.bfloat16,  # 設定浮點數精度為 bfloat16
    "device_map": "auto",  # 自動選擇設備 (例如 GPU)
}

# 載入模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 設置模型為評估模式
model.eval()

def generate_answer(question, model, tokenizer, max_length=200, num_return_sequences=1):
    # 編碼輸入問題
    inputs = tokenizer.encode(question, return_tensors='pt').to(model.device)
    
    # 生成答案
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences, do_sample=True)
    
    # 解碼生成的答案
    answers = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return answers

# 示例問題
question = "Create a suspenseful story about a man who wakes up in a strange hotel room with no memory of how he got there."

# 生成答案
answers = generate_answer(question, model, tokenizer)

# 輸出答案
for idx, answer in enumerate(answers):
    print(f"Answer {idx+1}: {answer}")
