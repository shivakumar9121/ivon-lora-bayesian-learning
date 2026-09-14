"""Qwen option-token classification with LoRA on q_proj and v_proj."""
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


def option_token_ids(tokenizer):
    ids = [tokenizer.encode(" " + letter, add_special_tokens=False) for letter in "ABCD"]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != 4:
        raise ValueError("The tokenizer must encode ' A', ' B', ' C', ' D' as four single tokens")
    return [x[0] for x in ids]


def attach_lora(base, config):
    model = get_peft_model(base, LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], bias="none", task_type="CAUSAL_LM",
        target_modules=config["target_modules"]))
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected trainable base parameter: {name}")
            p.data = p.data.float()
    model.config.use_cache = False
    return model


def option_logits(model, batch, token_ids):
    base = model.get_base_model()
    # Qwen2Model contains the injected LoRA layers. Project only the final hidden
    # state onto four frozen vocabulary rows instead of allocating B*L*152k logits.
    # This is algebraically identical to selecting these logits from the full LM.
    if base.config.model_type != "qwen2":
        raise ValueError("This memory-saving path is validated for Qwen2 only")
    out = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                     use_cache=False, return_dict=True)
    last = batch["attention_mask"].sum(1) - 1
    hidden = out.last_hidden_state[torch.arange(len(last), device=last.device), last]
    return F.linear(hidden, base.lm_head.weight[token_ids]).float()
