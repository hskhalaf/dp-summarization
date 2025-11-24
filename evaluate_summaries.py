import os, json, argparse, re
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer


def build_reference(ex):
    return ex.get("reference","") or ""

def rouge_scores(pred, ref):
    if not ref:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    s = scorer.score(ref, pred)
    return {"rouge1": s["rouge1"].fmeasure, "rouge2": s["rouge2"].fmeasure, "rougeL": s["rougeL"].fmeasure}

def render_chat(tokenizer, messages, add_generation_prompt=True):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

def judge_prompt(reference, candidate):
    return [
        {"role": "system",
         "content": "You are an impartial evaluator. Output ONLY a single JSON object with integer fields."},
        {"role": "user",
         "content":
            "Reference (gold):\n" + reference + "\n\n"
            "Candidate:\n" + candidate + "\n\n"
            "Rate coverage (does the candidate include key points from the reference), faithfulness (no contradictions), "
            "and clarity (writing quality) on 1-10. Also include an overall 1-10 rounded average.\n"
            'Return JSON exactly like {"coverage": X, "faithfulness": Y, "clarity": Z, "overall": W}.'
        }
    ]

@torch.no_grad()
def llm_judge(model, tokenizer, reference: str, candidate: str, device="cpu"):
    if not reference or not candidate:
        return {"coverage": 0, "faithfulness": 0, "clarity": 0, "overall": 0}
    prompt = render_chat(tokenizer, judge_prompt(reference, candidate), add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs, do_sample=False, max_new_tokens=128,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return {"coverage": 0, "faithfulness": 0, "clarity": 0, "overall": 0}
    try:
        obj = json.loads(m.group(0))
        def get_int(k): 
            try: return int(obj.get(k, 0))
            except: return 0
        return {
            "coverage": get_int("coverage"),
            "faithfulness": get_int("faithfulness"),
            "clarity": get_int("clarity"),
            "overall": get_int("overall"),
        }
    except Exception:
        return {"coverage": 0, "faithfulness": 0, "clarity": 0, "overall": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", type=str, required=True, help="outputs.json from make_summaries.py")
    ap.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--judge", action="store_true", help="run LLM judge as well")
    ap.add_argument("--out_json", type=str, default="eval_results.json")
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    outputs = data.get("outputs", [])

    tokenizer = None; model = None
    if args.judge:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model.to(args.device); model.eval()

    per = []
    agg = {
        "non_dp": {"rouge1": [], "rouge2": [], "rougeL": []},
        "dp": {"rouge1": [], "rouge2": [], "rougeL": []},
        "judge_non_dp": {"coverage": [], "faithfulness": [], "clarity": [], "overall": []},
        "judge_dp": {"coverage": [], "faithfulness": [], "clarity": [], "overall": []},
    }

    for ex in tqdm(outputs, desc="Evaluating"):
        ref = build_reference(ex)
        s_ndp = ex.get("non_dp_summary","")
        s_dp  = ex.get("dp_summary","")

        r_ndp = rouge_scores(s_ndp, ref)
        r_dp  = rouge_scores(s_dp, ref)
        for k in ["rouge1","rouge2","rougeL"]:
            agg["non_dp"][k].append(r_ndp[k])
            agg["dp"][k].append(r_dp[k])

        j_ndp = j_dp = None
        if args.judge:
            j_ndp = llm_judge(model, tokenizer, ref, s_ndp, device=args.device)
            j_dp  = llm_judge(model, tokenizer, ref, s_dp, device=args.device)
            for k in ["coverage","faithfulness","clarity","overall"]:
                agg["judge_non_dp"][k].append(j_ndp[k])
                agg["judge_dp"][k].append(j_dp[k])

        per.append({
            "example_id": ex.get("example_id",""),
            "num_reviews": ex.get("num_reviews",0),
            "rouge_non_dp": r_ndp,
            "rouge_dp": r_dp,
            "judge_non_dp": j_ndp,
            "judge_dp": j_dp,
            "dp_eps_spent": ex.get("dp_eps_spent", 0.0),
        })

    def avg(lst):
        return sum(lst) / max(1, len(lst))

    aggregates = {
        "rouge_non_dp": {k: avg(v) for k, v in agg["non_dp"].items()},
        "rouge_dp": {k: avg(v) for k, v in agg["dp"].items()},
        "judge_non_dp": {k: avg(v) for k, v in agg["judge_non_dp"].items()} if args.judge else None,
        "judge_dp": {k: avg(v) for k, v in agg["judge_dp"].items()} if args.judge else None,
    }

    out = {"in_file": args.in_json, "aggregates": aggregates, "per_example": per}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", args.out_json)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
