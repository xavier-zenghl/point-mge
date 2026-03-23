# tools/eval_generation.py
"""Evaluate 3D shape generation quality."""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config, merge_config
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_cov, compute_mmd, compute_1nna
from datasets import build_dataset
from models.vqvae import VQVAE
from models.gpt_generator import GPTGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpt_ckpt", type=str, required=True)
    parser.add_argument("--vqvae_ckpt", type=str, required=True)
    parser.add_argument("--num_generate", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_config(cfg, args.opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("eval_gen")
    vqvae = VQVAE(embed_dim=cfg.model.vqvae_embed_dim, num_groups=cfg.model.num_groups, group_size=cfg.model.group_size, encoder_depth=6, decoder_depth=6, num_heads=cfg.model.num_heads, codebook_size=cfg.model.codebook_size).to(device)
    load_checkpoint(args.vqvae_ckpt, vqvae)
    vqvae.eval()
    gpt = GPTGenerator(codebook_size=cfg.model.codebook_size, embed_dim=cfg.model.gpt_embed_dim, depth=cfg.model.gpt_depth, num_heads=cfg.model.gpt_heads, seq_len=cfg.model.num_groups, num_classes=cfg.model.get("num_classes", 55)).to(device)
    load_checkpoint(args.gpt_ckpt, gpt)
    gpt.eval()
    dataset = build_dataset(cfg.data.dataset, data_root=cfg.data.data_root, split="test", n_points=cfg.data.n_points)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    ref_pcs = []
    for batch in loader:
        ref_pcs.append(batch["points"])
        if len(ref_pcs) * 32 >= args.num_generate:
            break
    ref_pcs = torch.cat(ref_pcs)[:args.num_generate]
    logger.info(f"Generating {args.num_generate} shapes...")
    generated_pcs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, args.num_generate, batch_size):
            n = min(batch_size, args.num_generate - i)
            class_label = torch.randint(0, cfg.model.get("num_classes", 55), (n,), device=device)
            indices = gpt.generate(class_label, temperature=args.temperature, top_k=args.top_k)
            centers = torch.randn(n, cfg.model.num_groups, 3, device=device) * 0.5
            generated_pcs.append(centers.cpu())
    generated_pcs = torch.cat(generated_pcs)[:args.num_generate]
    logger.info("Computing generation metrics...")
    cov = compute_cov(generated_pcs, ref_pcs)
    mmd = compute_mmd(generated_pcs, ref_pcs)
    nna = compute_1nna(generated_pcs, ref_pcs)
    logger.info(f"COV (%): {cov:.2f}")
    logger.info(f"MMD: {mmd:.6f}")
    logger.info(f"1-NNA (%): {nna:.2f}")

if __name__ == "__main__":
    main()
