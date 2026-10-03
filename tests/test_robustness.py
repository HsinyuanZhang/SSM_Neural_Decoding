import torch
from ssm_decode import ModelConfig, build_model
from ssm_decode.robustness import apply_persistent_weight_noise
from ssm_decode.hardware import calibrate_scale, fake_quantize


def test_persistent_noise_is_deterministic():
    m=build_model(ModelConfig(3,2,width=4)); a=apply_persistent_weight_noise(m,.02,9); b=apply_persistent_weight_noise(m,.02,9)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))


def test_scale_is_frozen_and_needs_no_labels():
    support=torch.tensor([[-1.,2.],[.5,-.2]])
    scale=calibrate_scale(support,8); q1,_=fake_quantize(torch.tensor([[4.,-4.]]),8,scale); q2,_=fake_quantize(torch.tensor([[4.,-4.]]),8,scale)
    assert torch.equal(q1,q2) and scale > 0
