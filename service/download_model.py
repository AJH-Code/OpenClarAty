#!/usr/bin/env python3
"""
Download the CLaRa compression-128 model from HuggingFace.

This downloads Apple's CLaRa-7B-E2E model with 128x compression.
The base Mistral-7B-Instruct-v0.2 model will be downloaded automatically
on first use by the transformers library.

Usage:
    python download_model.py [--output ./models/compression-128]
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Download CLaRa model")
    parser.add_argument("--output", default="./models/compression-128",
                        help="Output directory for the model")
    parser.add_argument("--repo", default="apple/CLaRa-7B-E2E",
                        help="HuggingFace repo ID")
    parser.add_argument("--subfolder", default="compression-128",
                        help="Subfolder within the repo")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    output_dir = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    print(f"Downloading CLaRa model from {args.repo}/{args.subfolder}...")
    print(f"Output: {output_dir}")
    print()

    # Download the specific subfolder
    snapshot_download(
        repo_id=args.repo,
        local_dir=output_dir,
        allow_patterns=[f"{args.subfolder}/**"],
    )

    print(f"\n✅ Model downloaded to: {output_dir}")
    print(f"\nNote: The base Mistral-7B-Instruct-v0.2 model (~14GB) will be")
    print(f"downloaded automatically from HuggingFace on first service start.")
    print(f"This is cached in ~/.cache/huggingface/hub/")


if __name__ == "__main__":
    main()
