import os
import json
import glob
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_json_summary(text):
    text = text.strip()
    start = text.find('{')
    end = text.rfind('}') + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"raw": text}

@torch.inference_mode()
def generate_batch(model, tokenizer, reviews_list, max_new_tokens, device, temperature, top_p, ctx_chars):
    prompts = []
    for reviews in reviews_list:
        joined = "\n\n".join(reviews)[-ctx_chars:]
        messages = [
            {"role": "system", "content": "You are a precise, neutral summarizer of customer reviews. Generate a JSON object with three fields: 'verdict' (one sentence overall summary), 'pros' (array of positive aspects), and 'cons' (array of negative aspects). Return only valid JSON, no additional text."},
            {"role": "user", "content": f"Context:\n{joined}\n\nGenerate a JSON summary with verdict, pros, and cons:"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(**inputs, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
    summaries = []

    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        generated_ids = output[input_len:]
        summary_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        summary = parse_json_summary(summary_text)
        summaries.append(summary)
    
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Generate non-DP summaries of customer reviews")
    parser.add_argument("--data_root", type=str, default="summary_data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--ctx_chars", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--out_json", type=str, default="outputs_nodp.json")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16 if device.startswith("cuda") or device == "mps" else torch.float32
    
    print(f"Loading model: {args.model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    split_dir = os.path.join(args.data_root, args.split)
    files = sorted(glob.glob(os.path.join(split_dir, "*.json")))[:args.max_examples]
    
    print(f"Processing {len(files)} file(s) with batch_size={args.batch_size}...")

    all_reviews = []
    all_refs = []
    example_ids = []
    
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            item = json.load(f)

        reviews = []
        for r in item.get("customer_reviews", []):
            txt = ((r.get("title") or "") + "\n" + (r.get("text") or "")).strip()
            if txt: reviews.append(txt)
        
        refs = []
        for s in item.get("website_summaries", []):
            if s.get("verdict"): refs.append(str(s["verdict"]))
            if s.get("pros"): refs.append("Pros: " + "; ".join(s.get("pros", [])))
            if s.get("cons"): refs.append("Cons: " + "; ".join(s.get("cons", [])))
        
        if reviews:
            all_reviews.append(reviews)
            all_refs.append("\n".join(refs).strip())
            example_ids.append(os.path.basename(fp))
    
    results = []
    for i in tqdm(range(0, len(all_reviews), args.batch_size), desc="Generating summaries"):
        batch_reviews = all_reviews[i:i+args.batch_size]
        summaries = generate_batch(model, tokenizer, batch_reviews, args.max_new_tokens, device, args.temperature, args.top_p, args.ctx_chars)
        
        for j, summary in enumerate(summaries):
            results.append({
                "example_id": example_ids[i+j],
                "num_reviews": len(batch_reviews[j]),
                "reference": all_refs[i+j],
                "summary": summary
            })
    
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "model": args.model_name,
            "device": device,
            "outputs": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(results)} summaries to {args.out_json}")

if __name__ == "__main__":
    main()
