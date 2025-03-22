from transformers import Gemma3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch

# 量子化の設定
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# モデルID
model_id = "google/gemma-3-4b-it"

# 量子化でモデルをロード
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,  # 量子化設定を適用
    device_map="cuda"
)

# プロセッサのロード
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

# チャット形式のメッセージ
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "AIの感情をテーマにした俳句を3つほど詠んで"}]
    }
]

# チャットテンプレートを適用してトークン化
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)  # dtypeを指定せず、量子化に任せる

input_len = inputs["input_ids"].shape[-1]

# 推論モードでテキスト生成
with torch.inference_mode():
    generation = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True
    )
    generation = generation[0][input_len:]

# デコードして出力
decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)