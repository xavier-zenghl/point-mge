# tests/test_gpt_generator.py
import torch
from models.gpt_generator import GPTGenerator

def test_gpt_generator_forward_shape():
    model = GPTGenerator(codebook_size=512, embed_dim=384, depth=12, num_heads=6, seq_len=64, num_classes=55)
    indices = torch.randint(0, 512, (2, 64))
    class_label = torch.randint(0, 55, (2,))
    logits = model(indices, class_label)
    assert logits.shape == (2, 64, 512)

def test_gpt_generator_unconditional():
    model = GPTGenerator(codebook_size=64, embed_dim=128, depth=4, num_heads=4, seq_len=16, num_classes=0)
    indices = torch.randint(0, 64, (2, 16))
    logits = model(indices)
    assert logits.shape == (2, 16, 64)

def test_gpt_generator_causal_mask():
    model = GPTGenerator(codebook_size=64, embed_dim=128, depth=2, num_heads=4, seq_len=8, num_classes=0)
    model.eval()
    indices = torch.randint(0, 64, (1, 8))
    logits1 = model(indices)
    indices_modified = indices.clone()
    indices_modified[0, 5] = (indices[0, 5] + 1) % 64
    logits2 = model(indices_modified)
    assert torch.allclose(logits1[:, :5], logits2[:, :5], atol=1e-5)
    assert not torch.allclose(logits1[:, 5:], logits2[:, 5:])

def test_gpt_generator_generate():
    model = GPTGenerator(codebook_size=64, embed_dim=128, depth=2, num_heads=4, seq_len=16, num_classes=10)
    model.eval()
    class_label = torch.tensor([3])
    generated = model.generate(class_label, temperature=1.0, top_k=10)
    assert generated.shape == (1, 16)
    assert generated.min() >= 0
    assert generated.max() < 64

def test_gpt_generator_gradient():
    model = GPTGenerator(codebook_size=64, embed_dim=128, depth=2, num_heads=4, seq_len=16, num_classes=10)
    indices = torch.randint(0, 64, (2, 16))
    labels = torch.tensor([0, 5])
    logits = model(indices, labels)
    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break
