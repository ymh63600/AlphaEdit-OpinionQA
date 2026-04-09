import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class OpinionQADataset(Dataset):
    

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, use_persona=False, data_rel_path=None, *args, **kwargs):

        if data_rel_path is None:
            data_rel_path = "edit_qkey_101224.0_withOp.json"


        data_path = Path(data_dir) / "opinionQA" / data_rel_path
        
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset Not Found：{data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        # With Persona
        persona_prefix = ""
        if use_persona:
            raw_metadata = raw_json.get("metadata", {})

            target_keys = [
                'CREGION', 'AGE', 'SEX', 'RACE', 'CITIZEN', 'MARITAL', 
                'RELIG', 'EDUCATION', 'POLPARTY', 'INCOME', 'RELIGATTEND', 'POLIDEOLOGY'
            ]

            persona_items = [
                f"{k}: {raw_metadata[k]}" 
                for k in target_keys if k in raw_metadata
            ]
            persona_str = ", ".join(persona_items)
            
            persona_prefix = (
                f"You are a survey respondent with the following profile: {persona_str}.\n"
                f"Please answer the survey question based on this profile.\n"
            )

        data = []
        entries = raw_json.get("entries", [])
        
        for i, record in enumerate(entries):

            prompt_template = record.get("prompt", "{}")
            subject = record.get("subject", "")
            target_new = record.get("target", "")
            target_true = record.get("target_true", "Unknown")

            
            final_prompt = f"{persona_prefix}Question: {prompt_template}\nAnswer:"
            # Paraphrase
            raw_rephrases = record.get("question_paraphrased", [])
            if isinstance(raw_rephrases, str):
                raw_rephrases = [raw_rephrases] if raw_rephrases else []
            
            if not raw_rephrases:
                # default prompt
                raw_rephrases = [prompt_template.format(subject)]
            
            paraphrase_prompts = [
                f"{persona_prefix}Question: {p}\nAnswer:"  
                for p in raw_rephrases
            ]
            
            # Implicit Questions
            raw_implicits = record.get("implicit_questions", []) 
            if isinstance(raw_implicits, str):
                raw_implicits = [raw_implicits] if raw_implicits else []
            
            generation_prompts = [
                f"{persona_prefix}Question: {p}\nAnswer:" 
                for p in raw_implicits
            ]

            # AlphaEdit format
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": final_prompt,
                        "subject": subject,
                        "target_new": {"str": target_new}, 
                        "target_true": {"str": target_true},
                    },
                    "paraphrase_prompts": paraphrase_prompts,
                    "neighborhood_prompts": [], 
                    "attribute_prompts": [],
                    "generation_prompts": generation_prompts,
                }
            )

        self._data = data[:size]
        print(f"Load Dataset (use_persona={use_persona})，Total Len {len(self._data)} 。")

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)