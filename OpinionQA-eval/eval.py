import torch
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Path configuration
model_id = "meta-llama/Llama-3.1-8B-Instruct"  # e.g., "meta-llama/Meta-Llama-3-8B"
delta_path = "model_delta_101224.pt"  # Path to the saved delta weights file

# 2. Load tokenizer and base model
print(f"--- Loading base model: {model_id} ---")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Use float16 or bfloat16 to save memory; device_map="auto" assigns GPUs automatically
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    device_map="auto",
    trust_remote_code=True
)

# 3. Manually apply delta weights to the model
def apply_delta(model, delta_pt_path):
    print(f"--- Applying delta weights: {delta_pt_path} ---")
    deltas = torch.load(delta_pt_path, map_location=model.device)
    
    with torch.no_grad():
        for param_name, delta in deltas.items():
            # Locate the corresponding parameter in the model by name
            # The keys in delta must match model.named_parameters()
            if param_name in model.state_dict():
                param = model.state_dict()[param_name]
                # W_new = W_old + Delta
                param.copy_(delta.to(param.device).to(param.dtype))
            else:
                print(f"Warning: Parameter {param_name} not found. Please check your hparams settings.")
    print("--- Weight update complete ---")

apply_delta(model, delta_path)
model.eval()

# 4. Define inference and saving function
def run_eval(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Skipping: file not found {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    print(f"Processing: {input_file}")
    
    for entry in tqdm(data['entries']):
        raw_prompt = entry['prompt']
        subject_text = entry.get('subject', '')
        target = entry['target']
        filled_prompt = raw_prompt.replace("{}", subject_text)

        # Construct a more constrained prompt
        # Adding "Please answer with only ONE of the options." is crucial
        structured_prompt = (
            f"You are a survey respondent.\n"
            f"Please answer with only ONE of the options.\n"
            f"Question:{filled_prompt}\n"
            f"Answer:"
        )

        inputs = tokenizer(structured_prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
        input_len = inputs.input_ids.shape[1] 

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20,  # Keep output concise by limiting to 20 tokens
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                # Force stopping at newline to avoid repetition issues
                eos_token_id=tokenizer.encode("\n", add_special_tokens=False)[-1] 
            )

        gen_ids = outputs[0][input_len:]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        print(f"Prompt: {structured_prompt}")
        print(f"Answer: {generated_text}")

        # Simple correctness check: whether target appears in the output
        is_correct = target.lower() in generated_text.lower()
        
        results.append({
            "case_id": entry.get("question_id", "N/A"),
            "prompt": structured_prompt,
            "target": target,
            "model_answer": generated_text,
            "is_correct": is_correct
        })

    # Wrap results back into original format and save
    output_payload = {
        "metadata": data.get("metadata", {}),
        "results": results,
        "summary": {
            "total": len(results),
            "accuracy": sum(1 for r in results if r["is_correct"]) / len(results) if results else 0
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_payload, f, indent=4, ensure_ascii=False)

    print(f"{input_file}.   {output_file}")
    print(f"Done! Accuracy: {output_payload['summary']['accuracy']:.2%}")

# 5. Run evaluation tasks
tasks = [
    ('edit_set_120_101224.json', 'eval_edit_resultso_101224.json'),
    ('test_same_topic_all_101224.json', 'eval_test_same_resultso_101224.json'),
    ('test_other_topics_101224.json', 'eval_test_other_resultso_101224.json')
]

for in_f, out_f in tasks:
    run_eval(in_f, out_f)