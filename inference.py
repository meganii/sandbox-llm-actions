import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def run_inference():
    # HF_TOKENを環境変数から取得
    hf_token = os.environ.get("HF_TOKEN")
    
    print("Hugging Face Token available:", bool(hf_token))
    print("Starting Gemma 3-4B-IT inference on CPU...")
    
    start_time = time.time()
    
    # モデルとトークナイザーの読み込み
    # 4bitの量子化設定
    model_id = "google/gemma-3-4b-it"
    
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    
    print(f"Loading model from {model_id}...")
    print(f"Current device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # CPU用の設定
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        token=hf_token
    )
    
    # 4bitの量子化を使うにはこちらの設定
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     load_in_4bit=True,
    #     token=hf_token
    # )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # 推論の実行
    prompts = [
        "東京の観光名所を5つ教えてください。",
        "機械学習とは何ですか？簡単に説明してください。",
        "良いプログラマーになるための5つのアドバイスをください。"
    ]
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # 推論開始時間の記録
        inference_start = time.time()
        
        # 入力のトークン化
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 推論の実行
        with torch.no_grad():
            generated_ids = model.generate(
                **input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # 結果のデコード
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # 推論終了時間の記録
        inference_time = time.time() - inference_start
        
        print(f"Response: {response}")
        print(f"Inference time: {inference_time:.2f} seconds")
        
        results.append({
            "prompt": prompt,
            "response": response,
            "inference_time": inference_time
        })
    
    # 結果の保存
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(f"Gemma 3-4B-IT Inference Results\n")
        f.write(f"Model load time: {load_time:.2f} seconds\n\n")
        
        for i, result in enumerate(results):
            f.write(f"=== Prompt {i+1} ===\n")
            f.write(f"{result['prompt']}\n\n")
            f.write(f"--- Response ---\n")
            f.write(f"{result['response']}\n\n")
            f.write(f"Inference time: {result['inference_time']:.2f} seconds\n\n")
        
        f.write(f"Total time: {time.time() - start_time:.2f} seconds\n")
    
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
    print("Results saved to results.txt")

if __name__ == "__main__":
    run_inference()