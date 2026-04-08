import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class OpinionQADataset(Dataset):
    """
    自定義調查問卷資料集，適配 AlphaEdit 框架，支援 Persona 注入。
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, use_persona=False, *args, **kwargs):
        # 1. 載入原始 JSON 檔案
        # data_path = Path(data_dir) / "opinionQA" / "edit_qkey_101224.0_withOp.json"
        # data_path = Path(data_dir) / "opinionQA" / "edit_set_120.json"
        # data_path = Path(data_dir) / "opinionQA" / "editing_requests_A_201501691138.0.json"
        # data_path = Path(data_dir) / "opinionQA" / "editing_requests_B_298722.0.json"
        # data_path = Path(data_dir) / "opinionQA" / "editing_requests_A_201501209397.0.json"
        # data_path = Path(data_dir) / "opinionQA" / "editing_requests_B_201501739617.0.json"
        # data_path = Path(data_dir) / "opinionQA" / "edit_set_120_112238.json"
        # data_path = Path(data_dir) / "opinionQA" / "edit_set_120_135822.json"
        # data_path = Path(data_dir) / "opinionQA" / "edit_set_120_155464.json"
        # data_path = Path(data_dir) / "opinionQA" / "edit_set_120_329777.json"
        # data_path = Path(data_dir) / "opinionQA" / "edit_set_120_205459.json"
        # data_path = Path(data_dir) / "opinionQA" / "edit_set_120_299715.json"
        # data_path = Path(data_dir) / "opinionQA" / "edit_set_120_600116.json"
        data_path = Path(data_dir) / "opinionQA" / "edit_set_120_692984.json"
        
        
        
        if not data_path.exists():
            raise FileNotFoundError(f"找不到問卷資料集：{data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        # 2. 處理 Persona (Metadata)
        persona_prefix = ""
        if use_persona:
            raw_metadata = raw_json.get("metadata", {})
            # 定義您指定的 12 個欄位
            target_keys = [
                'CREGION', 'AGE', 'SEX', 'RACE', 'CITIZEN', 'MARITAL', 
                'RELIG', 'EDUCATION', 'POLPARTY', 'INCOME', 'RELIGATTEND', 'POLIDEOLOGY'
            ]
            # 篩選並轉成 JSON 字串
            persona_items = [
                f"{k}: {raw_metadata[k]}" 
                for k in target_keys if k in raw_metadata
            ]
            persona_str = ", ".join(persona_items)
            
            # 建立 Prompt 前綴 (這裡沒有大括號了，所以不需要用 {{ }})
            persona_prefix = (
                f"You are a survey respondent with the following profile: {persona_str}.\n"
                f"Please answer the survey question based on this profile.\n"
            )

        data = []
        entries = raw_json.get("entries", [])
        
        for i, record in enumerate(entries):
            # 3. 處理基礎欄位
            prompt_template = record.get("prompt", "{}")
            subject = record.get("subject", "")
            target_new = record.get("target", "")
            target_true = record.get("target_true", "Unknown")

            # 4. 組合最終的 Prompt (加上 Persona 前綴與 Answer: 後綴)
            # AlphaEdit 在編輯時會用到這個 prompt
            final_prompt = f"{persona_prefix}Question: {prompt_template}\nAnswer:"
            print(final_prompt)
            # 處理 Paraphrase (驗證的一致性)
            raw_rephrases = record.get("question_paraphrased", [])
            if isinstance(raw_rephrases, str):
                raw_rephrases = [raw_rephrases] if raw_rephrases else []
            
            if not raw_rephrases:
                # 如果沒有改寫，就用格式化後的 prompt 當預設
                raw_rephrases = [prompt_template.format(subject)]
            
            paraphrase_prompts = [
                f"{persona_prefix}Question: {p}\nAnswer:"  
                for p in raw_rephrases
            ]
            
            # 處理 Implicit Questions (生成/泛化能力驗證)
            raw_implicits = record.get("implicit_questions", []) # 修正你的 key 為複數形式
            if isinstance(raw_implicits, str):
                raw_implicits = [raw_implicits] if raw_implicits else []
            
            generation_prompts = [
                f"{persona_prefix}Question: {p}\nAnswer:" 
                for p in raw_implicits
            ]

            # 5. 封裝成 AlphaEdit 要求的格式
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": final_prompt, # 這裡已包含 Persona
                        "subject": subject,
                        "target_new": {"str": target_new}, 
                        "target_true": {"str": target_true},
                    },
                    "paraphrase_prompts": paraphrase_prompts, # 這裡也包含 Persona
                    "neighborhood_prompts": [], 
                    "attribute_prompts": [],
                    "generation_prompts": generation_prompts, # 這裡也包含 Persona
                }
            )

        self._data = data[:size]
        print(f"成功載入資料集 (use_persona={use_persona})，共 {len(self._data)} 筆資料。")

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)