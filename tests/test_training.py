"""Offline checks on a randomly initialized tiny Qwen. These are not research results."""
import copy
import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from ivon_lora.data import Collator, format_arc
from ivon_lora.model import attach_lora, option_logits
from ivon_lora.train import make_optimizer, train_step, load_config


@pytest.fixture
def setup():
    torch.set_num_threads(2)
    torch.manual_seed(21)
    conf = load_config("configs/quick.json")
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64, attention_dropout=0.0))
    model = attach_lora(base,conf)
    batch = Collator(0)([{"input_ids":[2,3,4],"label":0},{"input_ids":[2,5],"label":2}])
    return model,batch,conf


def test_option_logits_match_full_model_with_padding(setup):
    model,batch,_ = setup
    model.eval()
    ids = [6,7,8,9]
    direct = option_logits(model,batch,ids)
    full = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],use_cache=False).logits
    selected = full[torch.arange(2),batch["attention_mask"].sum(1)-1][:,ids]
    torch.testing.assert_close(direct,selected,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize("method",["adamw","ivon"])
def test_only_adapters_change_and_posterior_restores(setup,method):
    model,batch,conf = setup
    frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(model,method,conf)
    loss = train_step(model,opt,[batch], [6,7,8,9],method)
    assert np.isfinite(loss)
    after = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    assert any(not torch.equal(a,b) for a,b in zip(before,after))
    for n,p in model.named_parameters():
        if n in frozen:
            torch.testing.assert_close(frozen[n],p,rtol=0,atol=0)
    if method == "ivon":
        with opt.sampled_params(train=False):
            assert any(not torch.equal(a,p) for a,p in zip(after,[p for p in model.parameters() if p.requires_grad]))
        for a,p in zip(after,[p for p in model.parameters() if p.requires_grad]):
            torch.testing.assert_close(a,p,rtol=0,atol=0)
        restored = make_optimizer(model,method,conf)
        restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        assert torch.all(restored.param_groups[0]["hess"] > 0)


def test_accumulated_gradient_matches_full_batch(setup):
    model,batch,conf = setup
    model2 = copy.deepcopy(model)
    opt1,opt2 = make_optimizer(model,"ivon",conf),make_optimizer(model2,"ivon",conf)
    torch.manual_seed(99)
    train_step(model,opt1,[batch],[6,7,8,9],"ivon")
    torch.manual_seed(99)
    train_step(model2,opt2,[{k:v[:1] for k,v in batch.items()},{k:v[1:] for k,v in batch.items()}],[6,7,8,9],"ivon")
    for a,b in zip(model.parameters(),model2.parameters()):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


def test_numeric_answer_keys_and_no_label_leak():
    row = {"id":"x","question":"Which option?","choices":{"label":["1","2","3","4"],"text":["one","two","three","four"]},"answerKey":"3"}
    item = format_arc(row)
    assert item["label"] == 2
    assert item["prompt"].endswith("Answer:")
    row["answerKey"] = "1"
    assert format_arc(row)["prompt"] == item["prompt"]
