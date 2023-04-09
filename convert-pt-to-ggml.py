import argparse
import os
import sys
import json
import struct
import numpy as np
import torch


def parse_args():

    parser = argparse.ArgumentParser(description='Convert a LLaMA model checkpoint to a ggml compatible file')
    parser.add_argument('model_name',  help='directory containing the model checkpoint')
    return parser.parse_args()


def load_hparams(model_name):

    # `dir_model` is something like `models/7B` or `models/7B/`.
    # "tokenizer.model" is expected under model's parent dir.
    # When `dir_model` is a symlink, f"{dir_model}/../tokenizer.model" would not be found.
    # Let's use the model's parent dir directly.
    fname_hparams = f"./model_configs/{model_name}.json"

    with open(fname_hparams, "r") as f:
        hparams = json.load(f)
        print(hparams)

    return hparams


def write_header(fout, hparams):

    vision_keys = ["image_size", "layers", "width", "head_width", "patch_size"]
    text_keys = ["context_length", "vocab_size", "width", "heads", "layers", "xattn", "fusedLN"]
    values = [
        0x67676d66,  # magic: ggmf in hex
        1, # file version
        hparams["embed_dim"],
        *[hparams["vision_cfg"][key] for key in vision_keys],
        *[int(hparams["text_cfg"][key]) for key in text_keys],
    ]
    fout.write(struct.pack("i" * len(values), *values))
    mlp_ratio = hparams["vision_cfg"]["mlp_ratio"]
    fout.write(struct.pack("f", mlp_ratio))


def process_and_write_variables(fout, model):

    for name, datao in model.items():

        if name.endswith("freqs"):
            continue

        shape = datao.shape

        print(f"Processing variable: {name} with shape: {shape} and type: {datao.dtype}")

        data = datao.numpy().squeeze()

        # header
        sname = name.encode('utf-8')
        fout.write(struct.pack("ii", len(data.shape), len(sname)))
        # for dim in reversed(data.shape):
        for dim in data.shape:
            fout.write(struct.pack("i", dim))
        fout.write(sname)

        # data output to file
        data.tofile(fout)


def main():

    args = parse_args()
    model_name = args.model_name  # EVA02-CLIP-B-16
    
    hparams = load_hparams(model_name)

    print(args)

    model_dir = f"./models/{model_name}"
    files= os.listdir(model_dir) 
    for fname_model in files: 
        if os.path.splitext(fname_model)[-1] =='.pt':
            break
    fname_model = os.path.join(model_dir,fname_model)
    fname_out = os.path.join(model_dir,f"ggml-model-f16.bin")

    model = torch.load(fname_model, map_location="cpu")

    with open(fname_out, "wb") as fout:
        write_header(fout, hparams)
        process_and_write_variables(fout, model)

    del model

    print(f"Done. Output file: {fname_out}")

if __name__ == "__main__":
    main()
