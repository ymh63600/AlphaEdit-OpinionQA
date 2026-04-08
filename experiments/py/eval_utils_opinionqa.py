import typing
from itertools import chain
import numpy as np
import torch
from experiments.py.eval_utils_counterfact import test_generation

def compute_rewrite_quality_opinionqa(
    model,
    tok,
    record: typing.Dict,
    snips=None, 
    vec=None,
) -> typing.Dict:
    # 1. 解構數據
    def ensure_plain_str(x):
        if isinstance(x, str): return x
        if isinstance(x, dict): return x.get("str", str(x))
        return str(x)

    subject = record["requested_rewrite"]["subject"]
    target_new = ensure_plain_str(record["requested_rewrite"]["target_new"])
    
    rewrite_prompts = [ensure_plain_str(record["requested_rewrite"]["prompt"].format(subject))]
    paraphrase_prompts = [ensure_plain_str(p) for p in record.get("paraphrase_prompts", [])]
    generation_prompts = [ensure_plain_str(p) for p in record.get("generation_prompts", [])]

    # --- zsRE 邏輯開始 ---
    
    # 2. 準備 Target Tokens (處理 Llama 的空格問題)
    # 在目標字串前加空格是為了符合自然生成的機率分佈
    target_tok = tok(" " + target_new)["input_ids"]
    if 'llama' in model.config._name_or_path.lower():
        # Llama tokenizer 會在開頭加一個 dummy prefix (BOS 或特定數值)
        # 我們只需要後面的實際內容
        target_tok = target_tok[1:]
    
    # 3. 定義 zsRE 核心預測函數
    def test_zsre_stability(prompts):
        results = []
        for prompt in prompts:
            is_step_correct = []
            # 模擬生成過程：從輸入 prompt 開始，逐個加入正確 token 並預測下一個
            for i in range(len(target_tok)):
                # 構造目前的前綴
                current_prefix = prompt + tok.decode(target_tok[:i])
                # 對 Llama 進行細微處理：非首字可能需要補回空格或保持 decode 原始狀態
                if 'llama' in model.config._name_or_path.lower() and i > 0:
                     # Llama decode 有時會丟失空格，確保銜接處正確
                     current_prefix = prompt + " " + tok.decode(target_tok[:i]).strip()

                inputs = tok(current_prefix, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    logits = model(**inputs).logits[:, -1, :] # 拿最後一個 token 的預測
                    pred_id = torch.argmax(logits, dim=-1).item()
                
                is_step_correct.append(pred_id == target_tok[i])
            
            # 只有當所有步驟都正確時，該 prompt 才算正確 (Exact Match)
            results.append(all(is_step_correct))
        return results

    # 4. 執行 zsRE 評估
    rewrite_correct = test_zsre_stability(rewrite_prompts)
    paraphrase_correct = test_zsre_stability(paraphrase_prompts)

    # --- 5. 執行 Generation 獲取實際回答 (保留原本需求) ---
    all_gen_inputs = rewrite_prompts + paraphrase_prompts + generation_prompts
    ret = {}

    if all_gen_inputs:
        try:
            rel_id = record["requested_rewrite"].get("relation_id", "default")
            consistency_texts = [x["text"] for x in snips[rel_id][record["requested_rewrite"].get("target_new_id", 0)]]
        except (KeyError, TypeError):
            consistency_texts = [target_new]

        gen_stats = test_generation(
            model,
            tok,
            all_gen_inputs,
            consistency_texts,
            [], 
            vec,
        )
        
        all_texts = gen_stats.get("text", [])
        n_rew = len(rewrite_prompts)
        n_para = len(paraphrase_prompts)

        # 這裡就是你要的：保留當時模型真正的回答文字
        ret["gen_original"] = all_texts[:n_rew]
        ret["gen_paraphrase"] = all_texts[n_rew : n_rew + n_para]
        ret["gen_implicit"] = all_texts[n_rew + n_para :]

        # 合併統計數據 (Entropy 等)
        ret.update({k: v for k, v in gen_stats.items() if k != "text"})

    # 6. 整合最終結果
    ret.update({
        "rewrite_prompts_correct": rewrite_correct,
        "paraphrase_prompts_correct": paraphrase_correct,
    })

    return ret