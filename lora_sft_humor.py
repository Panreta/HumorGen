
"""
Minimal LoRA SFT on a tiny humor dataset, using LoL-style templates
(Figure 2(a) selection and Figure 2(c) rewriting).

Goal: learn the mechanics (chat template, label masking, LoRA, training loop),
not to produce a funny model. 10 examples is far too few for real humor.

Runs on CUDA (Colab T4), Apple Silicon (MPS), or CPU (slow but works).
"""

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN = 256


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# 1. Data: (prompt, response) pairs in LoL's template formats
# ---------------------------------------------------------------------------
SELECT = (
    "Question: {q}\nAnswers: 1. {a1} 2. {a2}\n"
    "Which answer is more humorous? What is the humorous logic?"
)
REWRITE = (
    "Question: {q}\nAnswer: {a}\n"
    "Please rewrite this common answer into an interesting or humorous one."
)

DATA = [
    # Figure 2(a)-style selection tasks
    {"prompt": SELECT.format(q="Why was the cookie sad?",
                             a1="Because it got dropped on the floor.",
                             a2="Because his mom was a wafer so long."),
     "response": "Answer 2. \"A wafer so long\" sounds like \"away for so long\": "
                 "the cookie misses its mom, told through a cookie pun."},
    {"prompt": SELECT.format(q="Why did the student bring a ladder to class?",
                             a1="Because they wanted to go to high school.",
                             a2="To reach the top shelf."),
     "response": "Answer 1. \"High school\" is read literally: a ladder makes you higher."},
    {"prompt": SELECT.format(q="Why do programmers prefer dark mode?",
                             a1="It is easier on the eyes.",
                             a2="Because light attracts bugs."),
     "response": "Answer 2. Real bugs fly toward light; software bugs are the programmer's enemy."},
    {"prompt": SELECT.format(q="What did the ocean say to the beach?",
                             a1="Nothing, it just waved.",
                             a2="It said hello."),
     "response": "Answer 1. \"Waved\" means both a greeting and the motion of water."},
    # Figure 2(c)-style rewriting tasks
    {"prompt": REWRITE.format(q="What does a clock do when it's hungry?",
                              a="It eats some food."),
     "response": "It goes back for seconds."},
    {"prompt": REWRITE.format(q="Why don't skeletons fight each other?",
                              a="Because they are not alive."),
     "response": "They don't have the guts."},
    {"prompt": REWRITE.format(q="What do you call a fake noodle?",
                              a="A noodle that is not real."),
     "response": "An impasta."},
    {"prompt": REWRITE.format(q="Why did the math book look sad?",
                              a="Because nobody read it."),
     "response": "It had too many problems."},
    {"prompt": REWRITE.format(q="Why can't a bicycle stand on its own?",
                              a="Because it has no kickstand."),
     "response": "It's two tired."},
    {"prompt": REWRITE.format(q="Why did the scarecrow win an award?",
                              a="Because he did a good job."),
     "response": "He was outstanding in his field."},
]

# Held-out prompt, not in the training data
TEST_PROMPT = REWRITE.format(q="Why did the coffee call the police?",
                             a="Because someone took it.")


# ---------------------------------------------------------------------------
# 2. Tokenization with label masking: loss only on the response tokens
# ---------------------------------------------------------------------------
def build_example(ex, tok):
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": ex["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,  # appends "<|im_start|>assistant\n"
    )
    full_text = prompt_text + ex["response"] + tok.eos_token

    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"][:MAX_LEN]

    # -100 = "ignore this position" in PyTorch's cross-entropy
    labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids):])[:len(full_ids)]
    return {"input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels}


def make_collator(pad_id):
    def collate(batch):
        L = max(len(x["input_ids"]) for x in batch)
        def pad(seq, value):
            return seq + [value] * (L - len(seq))
        return {
            "input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in batch]),
            "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in batch]),
            "labels": torch.tensor([pad(x["labels"], -100) for x in batch]),
        }
    return collate


@torch.no_grad()
def generate(model, tok, prompt, device):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=40, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 3. Main: load -> baseline output -> wrap with LoRA -> train -> compare
# ---------------------------------------------------------------------------
def main():
    device = pick_device()
    print(f"Device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)

    print("\n=== BEFORE fine-tuning ===")
    print(generate(model, tok, TEST_PROMPT, device))

    lora_cfg = LoraConfig(
        r=8,                 # rank of the update matrices
        lora_alpha=16,       # update is scaled by alpha / r
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = Dataset.from_list(DATA).map(lambda ex: build_example(ex, tok),
                                     remove_columns=["prompt", "response"])

    args = TrainingArguments(
        output_dir="lora-humor-out",
        num_train_epochs=8,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,  # keep our custom "labels" column
    )
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=make_collator(tok.pad_token_id)).train()

    print("\n=== AFTER fine-tuning ===")
    print(generate(model, tok, TEST_PROMPT, device))

    model.save_pretrained("lora-humor-adapter")  # saves only the LoRA weights (a few MB)
    print("\nAdapter saved to ./lora-humor-adapter")


if __name__ == "__main__":
    main()