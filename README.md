# eva-02.cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Inference of [EVA-02-CLIP](https://arxiv.org/abs/2303.15389) model in pure C/C++

## Description

Inspired by the open-source project [llama.cpp](https://github.com/ggerganov/llama.cpp), we believe that similar approaches for implementing large language models (LLMs) using C/C++ on CPUs can also be applied to Transformer-based visual models, enabling the creation of multi-modal models for image-text tasks like CLIP.

As a result, we have implemented an improved variant of the CLIP model, named `EVA-02-CLIP` , which has been demonstrated to achieve better performance at smaller model sizes. With this project, you can perform zero-shot image classification on CPU using plain C/C++.

This project only support EVA02-CLIP-B-16 model now.

**Features:**

- Plain C/C++ implementation without dependencies
- Apple silicon first-class citizen - optimized via ARM NEON and Accelerate framework
- AVX2 support for x86 architectures
- Runs on the CPU

**Supported platforms:**

- [x] Mac OS
- [x] Linux
- [x] Windows (via CMake)

---

Here is a typical run using EVA02-CLIP-B-16:

```java
make -j && ./main -m ./models/EVA02-CLIP-B-16/ggml-model-f16.bin -i CLIP.png -c "a diagram,a dog,a cat"
I eva.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  x86_64
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512cd
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
I CXX:      g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512cd   -c ggml.c -o ggml.o
g++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread -c eva.cpp -o eva.o
g++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread -c utils.cpp -o utils.o
g++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread main.cpp ggml.o eva.o utils.o -o main 

====  Run ./main -h for help.  ====

eva_model_load: loading model from './models/EVA02-CLIP-B-16/ggml-model-f16.bin' - please wait ...
image height: 762, width: 2162
eva_model_load: n_embd = 512
eva_model_load: image_size = 224
eva_model_load: n_layers = 12
eva_model_load: width  = 768
eva_model_load: head_width  = 64
eva_model_load: patch_size  = 16
eva_model_load: context_length  = 77
eva_model_load: vocab_size = 49408
eva_model_load: width  = 512
eva_model_load: n_head  = 8
eva_model_load: n_layers = 12
eva_model_load: ggml ctx size = 286.24 MB
eva_model_load: loading model from './models/EVA02-CLIP-B-16/ggml-model-f16.bin'
eva_model_load: ...................................................... done
eva_model_load: model size =   286.14 MB / num tensors = 436

system_info: n_threads = 8 / 96 | AVX = 1 | AVX2 = 1 | AVX512 = 1 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 | 

probs:
0.943854
0.044589
0.011558

eva_print_timings:     load time =   322.75 ms
eva_print_timings:     eval time =  1158.96 ms
eva_print_timings:    total time =  1481.89 ms
```

## Usage

Here are the steps for the EVA02-CLIP-B-16 model.

### Get the Code

```bash
git clone https://github.com/Robert-zwr/eva-02.cpp
cd eva-02.cpp
```

### Build

- On Linux or MacOS:

  ```bash
  make
  ```

### Prepare Data & Run

You can download the original EVA02-CLIP-B-16 weights here:[🤗 HF link](https://huggingface.co/QuanSun/EVA-CLIP/blob/main/EVA02_CLIP_B_psz16_s8B.pt) (`286MB`)

```bash
# obtain the original EVA02-CLIP-B-16 model weight, and place it in ./models/EVA02-CLIP-B-16
mkdir -p models/EVA02-CLIP-B-16
cd models/EVA02-CLIP-B-16
wget https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_B_psz16_s8B.pt
cd ../..

# convert the EVA02-CLIP-B-16 model to ggml FP16 format
python3 convert-pt-to-ggml.py EVA02-CLIP-B-16

# run the inference
./main
```

When running the model, make sure you have enough disk space to store all the intermediate files.

The model size is about 286MB, and about 1600MB of memory will be pre-allocated during computation. So make sure your device has at least 2GB memory. 

## Details

Similar to the llama.cpp project, our project also relies on the [ggml](https://github.com/ggerganov/ggml) tensor library for computation, which is a machine learning inference framework based on directed computation graphs that is implemented in C language and is still under development. The supported tensor operations are currently limited. To adapt to the EVA02-CLIP-B-16 model, we have added support for the following operators:

+ 3D tensor broadcast
+ L2 normalization
+ conv2d
+ tensor slicing
+ tensor concatenating
+ 2D RoPE.

After testing, we found that if we use the same image preprocessing and text tokenization as the original code for inference, the results are very close to those obtained using [original Pytorch implementation](https://github.com/baaivision/EVA/tree/master/EVA-CLIP), indicating that our inference process is accurate. However, using C/C++ based image preprocessing and text tokenization for inference may result in slight deviations from the original results. Specifically:

- To avoid unnecessary third-party dependencies, we used the light-weight [stb](https://github.com/nothings/stb) image processing library for image resizing. However, this implementation process is not exactly the same as the underlying implementation logic of torchvision, resulting in slightly different image inputs of the model, but the difference is minimal.
- As we are not familiar with regular expressions in C++, the text normalization implemented in C++ in the project is not exactly equivalent to the text normalization implemented in the original Python code when dealing with special characters. At present, we only guarantee that the tokenization results obtained when processing most image labels (including the 1000 labels of the Imagenet-1K dataset) without special characters are the same as the original code.

For example, for the sample image "[CLIP.png](https://github.com/Robert-zwr/eva-02.cpp/blob/main/CLIP.png)" and the label "a diagram, a dog, a cat", the original code yields the probability of [0.860, 0.107, 0.033], while our C/C++ implementation yields the probability of [0.944, 0.045, 0.012]. However, we have not found any difference in image classification results caused by these deviations. Further analysis is needed to determine the specific impact of these differences on the Imagenet-1K dataset.

### TODO

- [ ] Get the accuracy on the Imagenet-1K
- [ ] Support more EVA02-CLIP models
- [ ] Support 4-bit integer quantization
