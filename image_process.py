from typing import Optional, Tuple
from torchvision.transforms import Normalize, Compose, ToTensor, Resize, \
    CenterCrop
from PIL import Image
import numpy as np
import struct

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def _convert_to_rgb(image):
    return image.convert('RGB')

def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    transforms = [
        Resize(image_size),
        CenterCrop(image_size),
        _convert_to_rgb,
        ToTensor(),
        Normalize(mean=mean, std=std)
    ]
    return Compose(transforms)

image_path = "CLIP.png"
preprocess = image_transform(
        224,
        is_train=False,
        mean=OPENAI_DATASET_MEAN,
        std=OPENAI_DATASET_STD,
    )
image = preprocess(Image.open(image_path)).unsqueeze(0)
image = image[0].numpy()
print(image.shape)

fname_out = "/home/zwr/EVA_env/eva-02.cpp/temp/image.bin"
with open(fname_out, "wb") as fout:
    fout.write(struct.pack("i", len(image.shape)))
    for dim in reversed(image.shape):
        fout.write(struct.pack("i", dim))
    image.tofile(fout)