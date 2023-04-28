#include "eva.h"

#include "ggml.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"

#include <cinttypes>
#include <fstream>
#include <random>
#include <unordered_map>
#include <queue>
#include <regex>
#include <cassert>
#include <cstring>
#include <vector>
#include <algorithm>


struct vision_cfg {
    int32_t image_size = 224;
    int32_t layers = 12;
    int32_t width = 768;
    int32_t head_width = 64;
    int32_t patch_size = 16;
    float mlp_ratio = 2.6667;
};

struct text_cfg {
    int32_t context_length = 77;
    int32_t vocab_size = 49408;
    int32_t width = 512;
    int32_t heads = 8;
    int32_t layers = 12;
    bool xattn = true;
    bool fusedLN = true;
};


// default hparams for EVA02-CLIP-B-16
struct eva_hparams {
    int32_t n_embd = 512;
    vision_cfg vision_hparams;
    text_cfg text_hparams;
};

struct eva_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};

struct eva_vision_layer {
    // normalization
    struct ggml_tensor * norm1_weight;
    struct ggml_tensor * norm1_bias;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * bq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * bv;

    // normalization
    struct ggml_tensor * inner_attn_norm_weight;
    struct ggml_tensor * inner_attn_norm_bias;

    // projection
    struct ggml_tensor * wo;
    struct ggml_tensor * bo;

    //rope
    struct ggml_tensor * rope_cos;
    struct ggml_tensor * rope_sin;

    // normalization
    struct ggml_tensor * norm2_weight;
    struct ggml_tensor * norm2_bias;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * b1;
    struct ggml_tensor * w2;
    struct ggml_tensor * b2;
    struct ggml_tensor * ffn_norm_weight;
    struct ggml_tensor * ffn_norm_bias;
    struct ggml_tensor * w3;
    struct ggml_tensor * b3;
};

struct eva_text_layer {
    // normalization
    struct ggml_tensor * norm1_weight;
    struct ggml_tensor * norm1_bias;

    // attention
    struct ggml_tensor * in_proj_weight;
    struct ggml_tensor * in_proj_bias;
    struct ggml_tensor * out_proj_weight;
    struct ggml_tensor * out_proj_bias;

    // normalization
    struct ggml_tensor * norm2_weight;
    struct ggml_tensor * norm2_bias;

    //mlp
    struct ggml_tensor * c_fc_weight;
    struct ggml_tensor * c_fc_bias;
    struct ggml_tensor * c_proj_weight;
    struct ggml_tensor * c_proj_bias;
};

struct eva_vision_model {
    struct ggml_tensor * cls_token;
    struct ggml_tensor * pos_embeddings;
    struct ggml_tensor * patch_embed_weight;
    struct ggml_tensor * patch_embed_bias;
    struct ggml_tensor * rope_cos;
    struct ggml_tensor * rope_sin;
    std::vector<eva_vision_layer> layers;
    struct ggml_tensor * norm_weight;
    struct ggml_tensor * norm_bias;
    struct ggml_tensor * head_weight;
    struct ggml_tensor * head_bias;
};

struct eva_text_model {
    struct ggml_tensor * pos_embeddings;
    struct ggml_tensor * text_proj;
    struct ggml_tensor * token_embed;
    std::vector<eva_text_layer> layers;
    struct ggml_tensor * ln_final_weight;
    struct ggml_tensor * ln_final_bias;
};

struct eva_model {
    eva_hparams hparams;

    struct ggml_tensor * logit_scale;
    struct eva_vision_model vision_model;
    struct eva_text_model text_model;

    struct ggml_context * ctx;
    std::unordered_map<std::string, struct ggml_tensor *> tensors;
};

struct eva_vocab {
    using id    = int32_t;
    using token = std::string;

    std::unordered_map<token, id> encoder;
    std::map<std::vector<token>, int> bpe_ranks;
};

struct eva_context {
    std::mt19937 rng;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls

    eva_model model;
    eva_vocab vocab;

    struct ggml_tensor * image;
    struct ggml_tensor * text;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> label_probs;
};

//image_preprocess
static float* image_preprosess(std::string img_path, const int img_size){
    std::string src_name(img_path);
    int cols, rows, img_channels;
    int expected_channels = 3;
    const int image_size = img_size;

    unsigned char* raw_img = stbi_load(src_name.c_str(), &cols, &rows, &img_channels, STBI_rgb);
    printf("image height: %d, width: %d\n", rows, cols);

    //torchvision.transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC)
    int short_edge = std::min(rows, cols);
    float ratio = static_cast<float>(image_size) / short_edge;
    int resize_rows = rows * ratio;
    int resize_cols = cols * ratio;
    auto *resized_img = (unsigned char *) malloc(resize_rows * resize_cols * STBI_rgb);
    stbir_resize(raw_img, cols, rows, 0, resized_img, resize_cols, resize_rows, 0, STBIR_TYPE_UINT8, expected_channels, STBIR_ALPHA_CHANNEL_NONE, 0,
                 STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                 STBIR_FILTER_CATMULLROM, STBIR_FILTER_CATMULLROM,
                 STBIR_COLORSPACE_SRGB, nullptr);
    
    // torchvision.transforms.CenterCrop(image_size)
    int crop_width = image_size;
    int crop_height = image_size;
    // calulate crop position's upper-left coordinates
    int crop_x = (resize_cols - crop_width) / 2;
    int crop_y = (resize_rows - crop_height) / 2;
    unsigned char* cropped_img = new unsigned char[crop_width * crop_height * expected_channels];
    // crop the image
    for (int y = 0; y < crop_height; y++) {
        for (int x = 0; x < crop_width; x++) {
            int index = ((crop_y + y) * resize_cols + (crop_x + x)) * expected_channels;
            int crop_index = (y * crop_width + x) * expected_channels;
            for (int c = 0; c < expected_channels; c++) {
                cropped_img[crop_index + c] = resized_img[index + c];
            }
        }
    }

    //torchvision.transforms.ToTensor()
    int* int_image = new int[image_size*image_size*expected_channels];
    for (int i = 0; i < image_size * image_size; i++) {
        int_image[i] = (int)cropped_img[i * expected_channels];                           // R
        int_image[i+image_size*image_size] = (int)cropped_img[i*expected_channels+1];     // G
        int_image[i+image_size*image_size*2] = (int)cropped_img[i*expected_channels+2];   // B
    }

    //torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    float* float_image = new float[image_size*image_size*expected_channels];
    for (int i = 0; i < image_size * image_size; i++) {
        float_image[i] = (int_image[i] / 255.0 - 0.48145466) / 0.26862954;
        float_image[i + image_size * image_size] = (int_image[i + image_size * image_size] / 255.0 - 0.4578275) / 0.26130258;
        float_image[i + image_size * image_size * 2] = (int_image[i + image_size * image_size * 2] / 255.0 - 0.40821073) / 0.27577711;
    }

    stbi_image_free(raw_img);
    stbi_image_free(resized_img);
    stbi_image_free(cropped_img);
    delete[] int_image;

    return float_image;
}


std::string whitespace_clean_and_tolower(std::string text) {
    // Replace one or more whitespace characters with a single space character
    std::regex pattern("\\s+");
    std::string replacement(" ");
    std::string cleaned_text = std::regex_replace(text, pattern, replacement);
    
    // Remove leading and trailing whitespace characters
    cleaned_text.erase(0, cleaned_text.find_first_not_of(" "));
    cleaned_text.erase(cleaned_text.find_last_not_of(" ") + 1);

    std::transform(cleaned_text.begin(), cleaned_text.end(), cleaned_text.begin(), [](unsigned char c) { return std::tolower(c); });
    
    return cleaned_text;
}

std::vector<std::string> regex(const std::string& text) {
    std::vector<std::string> tokens;
    // Match the pattern using regex
    std::regex pattern(R"(([^\s()-]+)|([()-])|(-))");
    
    std::smatch match;
    std::string::const_iterator search_start(text.cbegin());
    while (std::regex_search(search_start, text.cend(), match, pattern)) {
        if (match[0].length() > 0) {
            //tokens.push_back(match[0]);
            const std::string temp = match[0];
            int len = temp.length();
            if (temp[len-1]=='s' && temp[len-2]=='\''){
                const std::string temp1 = temp.substr(0,len-2);
                tokens.push_back(temp1);
                tokens.push_back("\'s");
            }
            else if (temp[len-1]=='.'){
                const std::string temp1 = temp.substr(0,len-1);
                tokens.push_back(temp1);
                tokens.push_back(".");
            }
            else tokens.push_back(match[0]);
        }
        search_start = match.suffix().first;
    }
    return tokens;
}

// Return vector of symbol pairs in a word.
// Word is represented as vector of symbols (symbols being variable-length strings).
std::vector<std::vector<std::string>> get_pairs(const std::vector<std::string>& word) {
    std::vector<std::vector<std::string>> pairs;
    for (size_t i = 0; i < word.size() - 1; ++i) {
        std::vector<std::string> pair{word[i], word[i + 1]};
        pairs.push_back(pair);
    }
    return pairs;
}

// Apply byte pair encoding to token.
std::string bpe(std::string& token, const std::map<std::vector<std::string>, int>& bpe_ranks, std::map<std::string, std::string>& cache) {
    if (cache.find(token) != cache.end()) {
        return cache[token];
    }
    std::vector<std::string> word(token.size() - 1);
    for (int i = 0; i < (int)token.size() - 1; ++i) {
        word[i] = std::string(1, token[i]);
    }
    word.push_back(token.substr(token.size() - 1) + "</w>");
    std::vector<std::vector<std::string>> pairs = get_pairs(word);
    if (pairs.empty()) {
        return token + "</w>";
    }
    
    while (true) {
        std::vector<std::string> best_pair;
        int best_rank = INT_MAX;
        for (const auto& pair : pairs) {
            if (bpe_ranks.find(pair) != bpe_ranks.end() && bpe_ranks.at(pair) < best_rank) {
                best_pair = pair;
                best_rank = bpe_ranks.at(pair);
            }
        }
        if (best_pair.empty()) {
            break;
        }
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ++i) {
            if (i < word.size() - 1 && word[i] == best_pair[0] && word[i + 1] == best_pair[1]) {
                new_word.push_back(best_pair[0] + best_pair[1]);
                i += 1;
            } else {
                new_word.push_back(word[i]);
            }
        }
        word = new_word;
        if (word.size() == 1) {
            break;
        }
        pairs = get_pairs(word);
    }
    std::string new_token = "";
    for (const auto& s : word) {
        new_token += s + " ";
    }
    new_token.pop_back();  // Remove trailing space
    cache[token] = new_token;
    return new_token;
}

//
// model loading
//

static bool eva_model_load(
            const std::string & fname,
            const std::string & image_path,
            const std::string & label_text,
            eva_context & ectx)
    {
    fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    const int64_t t_start_us = ggml_time_us();

    ectx.t_start_us = t_start_us;

    std::vector<char> f_buf(1024*1024);

    auto & model = ectx.model;
    auto & image = ectx.image;
    auto & text  = ectx.text;
    auto & vocab = ectx.vocab;

    const int img_size = model.hparams.vision_hparams.image_size;
    static size_t image_buf_size = 2*img_size*img_size*3*ggml_type_sizef(GGML_TYPE_F32);
    static void * image_buf = malloc(image_buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/ image_buf_size,
        /*.mem_buffer =*/ image_buf,
    };
    struct ggml_context * ctx_src = ggml_init(params);
    image = ggml_new_tensor_3d(ctx_src, GGML_TYPE_F32, img_size, img_size, 3);
    auto preprocessed_image = image_preprosess(image_path, img_size);
    memcpy(image->data, preprocessed_image, ggml_nbytes(image));

    auto fin = std::ifstream(fname, std::ios::binary);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic == EVA_FILE_MAGIC_UNVERSIONED) {
            fprintf(stderr, "%s: invalid model file '%s' (too old, regenerate your model files!)\n",
                    __func__, fname.c_str());
            return false;
        }
        if (magic != EVA_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }

        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != EVA_FILE_VERSION) {
            fprintf(stderr, "%s: invalid model file '%s' (unsupported format version %" PRIu32 ", expected %d)\n",
                    __func__, fname.c_str(), format_version, EVA_FILE_VERSION);
            return false;
        }
    }

    //int n_ff = 0;
    auto & hparams = model.hparams;
    // load hparams
    {
        auto & vision_hparams = hparams.vision_hparams;
        auto & text_hparams = hparams.text_hparams;
        int32_t flag = 0;

        fin.read((char *) &hparams.n_embd, sizeof(hparams.n_embd));

        fin.read((char *) &vision_hparams.image_size,  sizeof(vision_hparams.image_size));
        fin.read((char *) &vision_hparams.layers,  sizeof(vision_hparams.layers));
        fin.read((char *) &vision_hparams.width,  sizeof(vision_hparams.width));
        fin.read((char *) &vision_hparams.head_width,  sizeof(vision_hparams.head_width));
        fin.read((char *) &vision_hparams.patch_size,  sizeof(vision_hparams.patch_size));

        fin.read((char *) &text_hparams.context_length,  sizeof(text_hparams.context_length));
        fin.read((char *) &text_hparams.vocab_size,  sizeof(text_hparams.vocab_size));
        fin.read((char *) &text_hparams.width,  sizeof(text_hparams.width));
        fin.read((char *) &text_hparams.heads,  sizeof(text_hparams.heads));
        fin.read((char *) &text_hparams.layers,  sizeof(text_hparams.layers));
        fin.read((char *) &flag,  sizeof(flag));
        text_hparams.xattn = (bool)flag;
        fin.read((char *) &flag,  sizeof(flag));
        text_hparams.fusedLN = (bool)flag;

        fin.read((char *) &vision_hparams.mlp_ratio, sizeof(vision_hparams.mlp_ratio));

        //hparams.n_ctx = n_ctx;

        //n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;

        fprintf(stderr, "%s: n_embd = %d\n", __func__, hparams.n_embd);
        fprintf(stderr, "%s: image_size = %d\n", __func__, vision_hparams.image_size);
        fprintf(stderr, "%s: n_layers = %d\n", __func__, vision_hparams.layers);
        fprintf(stderr, "%s: width  = %d\n", __func__, vision_hparams.width);
        fprintf(stderr, "%s: head_width  = %d\n", __func__, vision_hparams.head_width);
        fprintf(stderr, "%s: patch_size  = %d\n", __func__, vision_hparams.patch_size);

        fprintf(stderr, "%s: context_length  = %d\n", __func__, text_hparams.context_length);
        fprintf(stderr, "%s: vocab_size = %d\n", __func__, text_hparams.vocab_size);
        fprintf(stderr, "%s: width  = %d\n", __func__, text_hparams.width);
        fprintf(stderr, "%s: n_head  = %d\n", __func__, text_hparams.heads);
        fprintf(stderr, "%s: n_layers = %d\n", __func__, text_hparams.layers);
    }

    // load vocab
    {
        //load encoder
        std::string word;
        //vocab.encoder.resize(model.hparams.text_hparams.vocab_size);
        std::string first;
        std::string second;
        std::vector<char> tmp(64);

        for (int i = 0; i < model.hparams.text_hparams.vocab_size; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            if (len > 0) {
                tmp.resize(len);
                fin.read(tmp.data(), len);
                word.assign(tmp.data(), len);
            } else {
                word.clear();
            }
            vocab.encoder[word] = i;
            //fprintf(stderr, "%s: n_vocab = %d\n", __func__, i);
        }

        //load bpe_ranks
        for (int i = 0; i < 49152-256-2; i++) {
            uint32_t len;
            //first one in pair
            fin.read((char *) &len, sizeof(len));

            first.resize(len);
            if (len > 0) {
                tmp.resize(len);
                fin.read(tmp.data(), len);
                first.assign(tmp.data(), len);
            } else {
                first.clear();
            }

            //second one in pair
            fin.read((char *) &len, sizeof(len));

            second.resize(len);
            if (len > 0) {
                tmp.resize(len);
                fin.read(tmp.data(), len);
                second.assign(tmp.data(), len);
            } else {
                second.clear();
            }
            //vocab.encoder[word] = i;
            std::vector<std::string> pair = {first, second};
            vocab.bpe_ranks[pair] = i;
            //fprintf(stderr, "%s: n_pair = %d\n", __func__, i);
        }
    }

    //tokenize
    std::vector<std::string> labels;
    std::map<std::string, std::string> cache;
    {
        std::stringstream ss(label_text);
        std::string label;
        while (getline(ss, label, ',')) {
            labels.push_back(whitespace_clean_and_tolower(label));
        }

        text = ggml_new_tensor_2d(ctx_src, GGML_TYPE_I32, model.hparams.text_hparams.context_length, labels.size());
        int label_idx = 0;
        int bpe_token_idx = 0;

        for (auto& l : labels) {
            ((int*)(text->data))[(bpe_token_idx++)+hparams.text_hparams.context_length*label_idx] = 49406;  //'<start_of_text>'
            //((int*)(text->data))[(bpe_token_idx++)+hparams.text_hparams.context_length*label_idx] = 320;    //'a'
            //printf("%s\n", l.c_str());
            auto tokens = regex(l);
            for (auto& t : tokens) {
                t = bpe(t, vocab.bpe_ranks, cache);
                std::vector<std::string> bpe_tokens;
                std::stringstream sstream(t);
                std::string bpe_token;
                while (getline(sstream, bpe_token, ' ')) {
                    bpe_tokens.push_back(bpe_token);
                }
                for (auto& bpe_t : bpe_tokens) {
                    //printf("%s\n", bpe_t.c_str());
                    int idx = vocab.encoder[bpe_t];
                    //printf("%d\n", idx);
                    ((int*)(text->data))[(bpe_token_idx++)+hparams.text_hparams.context_length*label_idx] = idx;
                }
            }
            ((int*)(text->data))[bpe_token_idx+hparams.text_hparams.context_length*label_idx] = 49407;  //'<end_of_text>'
            label_idx++;
            bpe_token_idx = 0;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    // wtype is for per-layer weights, while vtype is for other weights
    ggml_type wtype, vtype;
    wtype = vtype = GGML_TYPE_F16;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;  //512
        //vision
        const int width   = hparams.vision_hparams.width;  //768
        const int image_size   = hparams.vision_hparams.image_size;  //224
        const int patch_size   = hparams.vision_hparams.patch_size;  //16
        const int patch_num   = image_size/patch_size;  //14
        const int head_width   = hparams.vision_hparams.head_width;  //64
        const int n_layers   = hparams.vision_hparams.layers;  //12
        const float mlp_ratio   = hparams.vision_hparams.mlp_ratio;  //2.6667
        const int n_mlp   = (int)(width*mlp_ratio);  //2048

        ctx_size += ggml_type_sizef(vtype); // logit_scale
        ctx_size += width*ggml_type_sizef(vtype); // visual.cls_token
        ctx_size += (patch_num*patch_num+1)*width*ggml_type_sizef(vtype); // visual.pos_embeddings

        ctx_size += (3*patch_size*patch_size*width+width)*ggml_type_sizef(vtype); // visual.patch_embed

        ctx_size += 2*patch_num*patch_num*head_width*ggml_type_sizef(vtype); // visual.rope


        //TrV
        ctx_size += n_layers*(2*width*ggml_type_sizef(wtype)); // norm1

        ctx_size += n_layers*((width*width+width)*ggml_type_sizef(wtype)); // wq
        ctx_size += n_layers*(width*width*ggml_type_sizef(wtype)); // wk
        ctx_size += n_layers*((width*width+width)*ggml_type_sizef(wtype)); // wv

        ctx_size += n_layers*(2*width*ggml_type_sizef(wtype)); // inner_attn_ln

        ctx_size += n_layers*((width*width+width)*ggml_type_sizef(wtype)); // proj
    
        ctx_size += n_layers*(2*patch_num*patch_num*head_width*ggml_type_sizef(wtype)); // rope
        
        ctx_size += n_layers*(2*width*ggml_type_sizef(wtype)); // norm2
        
        ctx_size += n_layers*((n_mlp*width+n_mlp)*ggml_type_sizef(wtype)); // w1
        ctx_size += n_layers*((n_mlp*width+n_mlp)*ggml_type_sizef(wtype)); // w2
        ctx_size += n_layers*(2*n_mlp*ggml_type_sizef(wtype)); // ffn_ln
        ctx_size += n_layers*((width*n_mlp+width)*ggml_type_sizef(wtype)); // w3

        ctx_size += 2*width*ggml_type_sizef(vtype); // final_norm
        ctx_size += (n_embd*width+n_embd)*ggml_type_sizef(vtype); // head

        ctx_size += (11 + 23*n_layers)*256; // object overhead


        //text
        const int context_length   = hparams.text_hparams.context_length;  //77
        const int vocab_size   = hparams.text_hparams.vocab_size;  //49408
        const int text_width   = hparams.text_hparams.width;  //512
        //const int n_head   = hparams.text_hparams.heads;  //8
        const int n_layer   = hparams.text_hparams.layers;  //12

        ctx_size += context_length*text_width*ggml_type_sizef(vtype); // positional_embedding
        //text.text_projection with shape: torch.Size([512, 512])
        ctx_size += (text_width*text_width)*ggml_type_sizef(vtype); // text_projection
        //text.token_embedding.weight with shape: torch.Size([49408, 512])
        ctx_size += (vocab_size*text_width)*ggml_type_sizef(vtype); // token_embedding

        //Transformer
        //text.transformer.resblocks.0.ln_1.weight with shape: torch.Size([512])
        ctx_size += n_layer*(2*text_width*ggml_type_sizef(wtype)); // norm1
        //text.transformer.resblocks.0.attn.in_proj_weight with shape: torch.Size([1536, 512])
        ctx_size += n_layer*((3*text_width*text_width+3*text_width)*ggml_type_sizef(wtype)); // in_proj
        //text.transformer.resblocks.0.attn.out_proj.weight with shape: torch.Size([512, 512]) 
        ctx_size += n_layer*((text_width*text_width+text_width)*ggml_type_sizef(wtype)); // out_proj
        //text.transformer.resblocks.0.ln_2.weight with shape: torch.Size([512])
        ctx_size += n_layer*(2*text_width*ggml_type_sizef(wtype)); // norm2
        //text.transformer.resblocks.0.mlp.c_fc.weight with shape: torch.Size([2048, 512])
        ctx_size += n_layer*((n_mlp*text_width+n_mlp)*ggml_type_sizef(wtype)); // mlp.c_fc
        //text.transformer.resblocks.0.mlp.c_proj.weight with shape: torch.Size([512, 2048])
        ctx_size += n_layer*((text_width*n_mlp+text_width)*ggml_type_sizef(wtype)); // mlp.c_proj

        //text.ln_final.weight with shape: torch.Size([512])
        ctx_size += 2*text_width*ggml_type_sizef(vtype); // final_norm
        ctx_size += (5 + 12*n_layer)*256; // object overhead

        fprintf(stderr, "%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;  //512
        model.logit_scale = ggml_new_tensor_1d(ctx, vtype, 1);
        model.tensors["logit_scale"] = model.logit_scale;

        //vision
        const int width   = hparams.vision_hparams.width;  //768
        const int image_size   = hparams.vision_hparams.image_size;  //224
        const int patch_size   = hparams.vision_hparams.patch_size;  //16
        const int patch_num   = image_size/patch_size;  //14
        const int head_width   = hparams.vision_hparams.head_width;  //64
        const int n_layers   = hparams.vision_hparams.layers;  //12
        const float mlp_ratio   = hparams.vision_hparams.mlp_ratio;  //2.6667

        model.vision_model.layers.resize(n_layers);

        model.vision_model.cls_token = ggml_new_tensor_1d(ctx, vtype, width);
        model.vision_model.pos_embeddings = ggml_new_tensor_2d(ctx, vtype, width, patch_num*patch_num+1);
        model.vision_model.patch_embed_weight = ggml_new_tensor_4d(ctx, vtype, patch_size, patch_size, 3, width);
        model.vision_model.patch_embed_bias = ggml_new_tensor_1d(ctx, vtype, width);
        model.vision_model.rope_cos = ggml_new_tensor_2d(ctx, vtype, head_width, patch_num*patch_num);
        model.vision_model.rope_sin = ggml_new_tensor_2d(ctx, vtype, head_width, patch_num*patch_num);


        // map by name
        model.tensors["visual.cls_token"] = model.vision_model.cls_token;
        model.tensors["visual.pos_embed"] = model.vision_model.pos_embeddings;
        model.tensors["visual.patch_embed.proj.weight"] = model.vision_model.patch_embed_weight;
        model.tensors["visual.patch_embed.proj.bias"] = model.vision_model.patch_embed_bias;
        model.tensors["visual.rope.freqs_cos"] = model.vision_model.rope_cos;
        model.tensors["visual.rope.freqs_sin"] = model.vision_model.rope_sin;

        for (int i = 0; i < n_layers; ++i) {
            auto & layer = model.vision_model.layers[i];

            layer.norm1_weight = ggml_new_tensor_1d(ctx, vtype, width);
            layer.norm1_bias   = ggml_new_tensor_1d(ctx, vtype, width);

            layer.wq = ggml_new_tensor_2d(ctx, wtype, width, width);
            layer.wk = ggml_new_tensor_2d(ctx, wtype, width, width);
            layer.wv = ggml_new_tensor_2d(ctx, wtype, width, width);
            layer.bq = ggml_new_tensor_1d(ctx, wtype, width);
            layer.bv = ggml_new_tensor_1d(ctx, wtype, width);

            layer.inner_attn_norm_weight = ggml_new_tensor_1d(ctx, wtype, width);
            layer.inner_attn_norm_bias   = ggml_new_tensor_1d(ctx, wtype, width);

            layer.wo = ggml_new_tensor_2d(ctx, wtype, width, width);
            layer.bo = ggml_new_tensor_1d(ctx, wtype, width);

            layer.rope_cos = ggml_new_tensor_2d(ctx, wtype, head_width, patch_num*patch_num);
            layer.rope_sin = ggml_new_tensor_2d(ctx, wtype, head_width, patch_num*patch_num);

            layer.norm2_weight = ggml_new_tensor_1d(ctx, wtype, width);
            layer.norm2_bias   = ggml_new_tensor_1d(ctx, wtype, width);

            layer.w1 = ggml_new_tensor_2d(ctx, wtype, width, width*mlp_ratio);
            layer.b1 = ggml_new_tensor_1d(ctx, wtype, width*mlp_ratio);
            layer.w2 = ggml_new_tensor_2d(ctx, wtype, width, width*mlp_ratio);
            layer.b2 = ggml_new_tensor_1d(ctx, wtype, width*mlp_ratio);
            layer.ffn_norm_weight = ggml_new_tensor_1d(ctx, wtype, width*mlp_ratio);
            layer.ffn_norm_bias   = ggml_new_tensor_1d(ctx, wtype, width*mlp_ratio);
            layer.w3 = ggml_new_tensor_2d(ctx, wtype, width*mlp_ratio, width);
            layer.b3 = ggml_new_tensor_1d(ctx, wtype, width);

            // map by name
            model.tensors["visual.blocks." + std::to_string(i) + ".norm1.weight"] = layer.norm1_weight;
            model.tensors["visual.blocks." + std::to_string(i) + ".norm1.bias"] = layer.norm1_bias;

            model.tensors["visual.blocks." + std::to_string(i) + ".attn.q_proj.weight"] = layer.wq;
            model.tensors["visual.blocks." + std::to_string(i) + ".attn.k_proj.weight"] = layer.wk;
            model.tensors["visual.blocks." + std::to_string(i) + ".attn.v_proj.weight"] = layer.wv;
            model.tensors["visual.blocks." + std::to_string(i) + ".attn.q_bias"] = layer.bq;
            model.tensors["visual.blocks." + std::to_string(i) + ".attn.v_bias"] = layer.bv;

            model.tensors["visual.blocks." + std::to_string(i) + ".attn.inner_attn_ln.weight"] = layer.inner_attn_norm_weight;
            model.tensors["visual.blocks." + std::to_string(i) + ".attn.inner_attn_ln.bias"] = layer.inner_attn_norm_bias;

            model.tensors["visual.blocks." + std::to_string(i) + ".attn.proj.weight"] = layer.wo;
            model.tensors["visual.blocks." + std::to_string(i) + ".attn.proj.bias"] = layer.bo;

            model.tensors["visual.blocks." + std::to_string(i) + ".attn.rope.freqs_cos"] = layer.rope_cos;
            model.tensors["visual.blocks." + std::to_string(i) + ".attn.rope.freqs_sin"] = layer.rope_sin;

            model.tensors["visual.blocks." + std::to_string(i) + ".norm2.weight"] = layer.norm2_weight;
            model.tensors["visual.blocks." + std::to_string(i) + ".norm2.bias"] = layer.norm2_bias;

            model.tensors["visual.blocks." + std::to_string(i) + ".mlp.w1.weight"] = layer.w1;
            model.tensors["visual.blocks." + std::to_string(i) + ".mlp.w1.bias"] = layer.b1;
            model.tensors["visual.blocks." + std::to_string(i) + ".mlp.w2.weight"] = layer.w2;
            model.tensors["visual.blocks." + std::to_string(i) + ".mlp.w2.bias"] = layer.b2;
            model.tensors["visual.blocks." + std::to_string(i) + ".mlp.ffn_ln.weight"] = layer.ffn_norm_weight;
            model.tensors["visual.blocks." + std::to_string(i) + ".mlp.ffn_ln.bias"] = layer.ffn_norm_bias;
            model.tensors["visual.blocks." + std::to_string(i) + ".mlp.w3.weight"] = layer.w3;
            model.tensors["visual.blocks." + std::to_string(i) + ".mlp.w3.bias"] = layer.b3;
        }

        model.vision_model.norm_weight = ggml_new_tensor_1d(ctx, vtype, width);
        model.vision_model.norm_bias   = ggml_new_tensor_1d(ctx, vtype, width);
        model.vision_model.head_weight = ggml_new_tensor_2d(ctx, vtype, width, n_embd);
        model.vision_model.head_bias   = ggml_new_tensor_1d(ctx, vtype, n_embd);

        model.tensors["visual.norm.weight"] = model.vision_model.norm_weight;
        model.tensors["visual.norm.bias"] = model.vision_model.norm_bias;
        model.tensors["visual.head.weight"] = model.vision_model.head_weight;
        model.tensors["visual.head.bias"] = model.vision_model.head_bias;


        //text
        const int context_length   = hparams.text_hparams.context_length;  //77
        const int vocab_size   = hparams.text_hparams.vocab_size;  //49408
        const int text_width   = hparams.text_hparams.width;  //512
        //const int n_head   = hparams.text_hparams.heads;  //8
        const int n_layer   = hparams.text_hparams.layers;  //12

        model.text_model.layers.resize(n_layer);

        model.text_model.pos_embeddings = ggml_new_tensor_2d(ctx, wtype, text_width, context_length);
        model.text_model.text_proj = ggml_new_tensor_2d(ctx, wtype, text_width, text_width);
        model.text_model.token_embed = ggml_new_tensor_2d(ctx, wtype, text_width, vocab_size);

        model.tensors["text.positional_embedding"] = model.text_model.pos_embeddings;
        model.tensors["text.text_projection"] = model.text_model.text_proj;
        model.tensors["text.token_embedding.weight"] = model.text_model.token_embed;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.text_model.layers[i];

            layer.norm1_weight = ggml_new_tensor_1d(ctx, vtype, text_width);
            layer.norm1_bias   = ggml_new_tensor_1d(ctx, vtype, text_width);
            layer.in_proj_weight = ggml_new_tensor_2d(ctx, vtype, text_width, 3*text_width);
            layer.in_proj_bias = ggml_new_tensor_1d(ctx, vtype, 3*text_width);
            layer.out_proj_weight = ggml_new_tensor_2d(ctx, vtype, text_width, text_width);
            layer.out_proj_bias = ggml_new_tensor_1d(ctx, vtype, text_width);
            layer.norm2_weight = ggml_new_tensor_1d(ctx, vtype, text_width);
            layer.norm2_bias   = ggml_new_tensor_1d(ctx, vtype, text_width);
            layer.c_fc_weight = ggml_new_tensor_2d(ctx, vtype, text_width, 4*text_width);
            layer.c_fc_bias = ggml_new_tensor_1d(ctx, vtype, 4*text_width);
            layer.c_proj_weight = ggml_new_tensor_2d(ctx, vtype, 4*text_width, text_width);
            layer.c_proj_bias = ggml_new_tensor_1d(ctx, vtype, text_width);

            // map by name
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".ln_1.weight"] = layer.norm1_weight;
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".ln_1.bias"] = layer.norm1_bias;
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".attn.in_proj_weight"] = layer.in_proj_weight;
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".attn.in_proj_bias"] = layer.in_proj_bias;
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".attn.out_proj.weight"] = layer.out_proj_weight;
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".attn.out_proj.bias"] = layer.out_proj_bias;
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".ln_2.weight"] = layer.norm2_weight;
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".ln_2.bias"] = layer.norm2_bias;
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".mlp.c_fc.weight"] = layer.c_fc_weight;
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".mlp.c_fc.bias"] = layer.c_fc_bias;
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".mlp.c_proj.weight"] = layer.c_proj_weight;
            model.tensors["text.transformer.resblocks." + std::to_string(i) + ".mlp.c_proj.bias"] = layer.c_proj_bias;
        }

        model.text_model.ln_final_weight = ggml_new_tensor_1d(ctx, vtype, text_width);
        model.text_model.ln_final_bias   = ggml_new_tensor_1d(ctx, vtype, text_width);

        model.tensors["text.ln_final.weight"] = model.text_model.ln_final_weight;
        model.tensors["text.ln_final.bias"] = model.text_model.ln_final_bias;
    }

    const size_t file_offset = fin.tellg();

    fin.close();

    std::vector<uint8_t> tmp;

    //const int part_id = i;
    //const int part_id = n_parts - i - 1;

    std::string fname_part = fname;

    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname_part.c_str());

    fin = std::ifstream(fname_part, std::ios::binary);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
    fin.seekg(file_offset);

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        fprintf(stderr, "%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            //int32_t ftype; //1

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            //fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[4] = { 1, 1, 1, 1};
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];

            
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            
            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2] || tensor->ne[3] != ne[3]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            size_t bpe = 0;

            bpe = ggml_type_size(GGML_TYPE_F16);

            //if (n_dims == 1 || n_parts == 1) {
            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }
            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            total_size += ggml_nbytes(tensor);

            //fprintf(stderr, "%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            if (++n_tensors % 8 == 0) {
                fprintf(stderr, ".");
                fflush(stderr);
            }
        }

        fprintf(stderr, " done\n");

        fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
    }

    fin.close();

    //lctx.logits.reserve(lctx.model.hparams.n_ctx);

    ectx.t_load_us = ggml_time_us() - t_start_us;

    return true;
}

// evaluate the transformer
//
//   - ectx:      eva context
//   - n_threads: number of threads to use
//
static bool eva_eval_internal(
        eva_context & ectx,
        const int   n_threads) {
    const int64_t t_start_us = ggml_time_us();

    //const int N = n_tokens;

    const auto & model   = ectx.model;
    const auto & hparams = model.hparams;
    const auto & vision_hparams = hparams.vision_hparams;
    const auto & text_hparams = hparams.text_hparams;

    const int n_embd  = hparams.n_embd; //512

    static size_t result_buf_size = (n_embd+1)*101*8;
    static void * result_buf = malloc(result_buf_size);

    struct ggml_init_params result_params = {
        /*.mem_size   =*/ result_buf_size,
        /*.mem_buffer =*/ result_buf,
    };

    ggml_cgraph result_gf = {};
    result_gf.n_threads = n_threads;

    struct ggml_context * ctx = ggml_init(result_params);
    if (!ctx) {
        fprintf(stderr, "%s: failed to result context\n", __func__);
        return false;
    }
    struct ggml_tensor * image_features = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

    const int image_size = vision_hparams.image_size;
    const int n_vision_layer = vision_hparams.layers;
    const int vision_width = vision_hparams.width;  //768
    const int head_width = vision_hparams.head_width; //64
    const int patch_size = vision_hparams.patch_size; //16
    const float ratio = vision_hparams.mlp_ratio; //2.6667

    const int patch_num   = image_size/patch_size;  //14
    const int n_head = vision_width/head_width; //12
    const int ratio_width = vision_width*ratio; //2048

    //vision tower
    {
    // TODO: fix this hardcoded size
    static size_t vision_buf_size = 512u*1024*1024*2;
    static void * vision_buf = malloc(vision_buf_size);

    struct ggml_init_params vision_params = {
        /*.mem_size   =*/ vision_buf_size,
        /*.mem_buffer =*/ vision_buf,
    };

    ggml_cgraph vision_gf = {};
    vision_gf.n_threads = n_threads;

    struct ggml_context * ctx0 = ggml_init(vision_params);
    //x = self.patch_embed(x)
    struct ggml_tensor * vision_proj_tmp = ggml_conv_2d(ctx0, model.vision_model.patch_embed_weight, ectx.image, vision_width, patch_num*patch_num);
    struct ggml_tensor * conv_bias = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
    for (int i = 0; i < vision_width; i++){
        *((float*)(conv_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.patch_embed_bias->data)+i));
    }
    struct ggml_tensor * conv_bias_broadcast = ggml_repeat(ctx0, conv_bias, vision_proj_tmp);
    struct ggml_tensor * vision_proj = ggml_add(ctx0, vision_proj_tmp, conv_bias_broadcast);
    //x = torch.cat((cls_tokens, x), dim=1)
    struct ggml_tensor * vision_cls_token = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
    for (int i = 0; i < vision_width; i++){
        *((float*)(vision_cls_token->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.cls_token->data)+i));
    }
    struct ggml_tensor * vision_proj_with_cls_token = ggml_cat(ctx0, vision_cls_token, vision_proj);
    //x = x + position_embeddings
    struct ggml_tensor * vision_pos_embeddings = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, vision_width, patch_num*patch_num+1);
    for (int i = 0; i < vision_width; i++){
        for (int j = 0; j < patch_num*patch_num+1; j++){
            *((float*)(vision_pos_embeddings->data)+j*vision_width+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.pos_embeddings->data)+j*vision_width+i));
        }
    }
    struct ggml_tensor * inpL = ggml_add(ctx0, vision_proj_with_cls_token, vision_pos_embeddings);

    //transformer layers
    for (int il = 0; il < n_vision_layer; ++il) {
        
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;
        //norm1
        cur = ggml_norm(ctx0, inpL);

        struct ggml_tensor * norm1_weight = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
        for (int i = 0; i < vision_width; i++){
            *((float*)(norm1_weight->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].norm1_weight->data)+i));
        }
        struct ggml_tensor * norm1_weight_broadcast = ggml_repeat(ctx0, norm1_weight, cur);
        cur = ggml_mul(ctx0, cur, norm1_weight_broadcast);

        struct ggml_tensor * norm1_bias = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
        for (int i = 0; i < vision_width; i++){
            *((float*)(norm1_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].norm1_bias->data)+i));
        }
        struct ggml_tensor * norm1_bias_broadcast = ggml_repeat(ctx0, norm1_bias, cur);
        cur = ggml_add(ctx0, cur, norm1_bias_broadcast);

        //prepare q,k,v for self-attention
        struct ggml_tensor * bq = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
        struct ggml_tensor * bv = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
        for (int i = 0; i < vision_width; i++){
            *((float*)(bq->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].bq->data)+i));
            *((float*)(bv->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].bv->data)+i));
        }
        struct ggml_tensor * bq_broadcast = ggml_repeat(ctx0, bq, cur);
        struct ggml_tensor * bv_broadcast = ggml_repeat(ctx0, bv, cur);

        struct ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.vision_model.layers[il].wq, cur), bq_broadcast);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.vision_model.layers[il].wk, cur);
        struct ggml_tensor * Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.vision_model.layers[il].wv, cur), bv_broadcast);

        // Q = Qcur.contiguous().view(N, n_head = n_head, head_width).permute(0, 2, 1, 3).continueous()
        struct ggml_tensor * Q =
            ggml_cpy(ctx0,
                ggml_permute(ctx0,
                    ggml_cpy(ctx0,
                            Qcur,
                            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_width, n_head, patch_num*patch_num+1)),
                0, 2, 1, 3),
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_width, patch_num*patch_num+1, n_head));  //# B, num_heads=12, N=197, C=64
        // K = Kcur.contiguous().view(N, n_head = n_head, head_width).permute(0, 2, 1, 3).continueous()
        struct ggml_tensor * K =
            ggml_cpy(ctx0,
                ggml_permute(ctx0,
                    ggml_cpy(ctx0,
                            Kcur,
                            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_width, n_head, patch_num*patch_num+1)),
                0, 2, 1, 3),
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_width, patch_num*patch_num+1, n_head));
        // V = Vcur.contiguous().view(N, n_head = n_head, head_width).permute(0, 2, 1, 3).continueous()
        struct ggml_tensor * V =
            ggml_cpy(ctx0,
                ggml_permute(ctx0,
                    ggml_cpy(ctx0,
                            Vcur,
                            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_width, n_head, patch_num*patch_num+1)),
                0, 2, 1, 3),
            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_width, patch_num*patch_num+1, n_head));

        struct ggml_tensor * Q_cls_token = ggml_split_get_first(ctx0, Q, 1, 1);
        struct ggml_tensor * Q_without_cls_token = ggml_split_get_second(ctx0, Q, 1, 1);
        struct ggml_tensor * K_cls_token = ggml_split_get_first(ctx0, K, 1, 1);
        struct ggml_tensor * K_without_cls_token = ggml_split_get_second(ctx0, K, 1, 1);
        
        //2D RoPE
        struct ggml_tensor * freqs_cos = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, head_width, patch_num*patch_num);
        struct ggml_tensor * freqs_sin = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, head_width, patch_num*patch_num);
        for (int i = 0; i < patch_num*patch_num; i++){
            for (int j = 0; j < head_width; j++){
                *((float*)(freqs_cos->data)+i*head_width+j) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.rope_cos->data)+i*head_width+j));
                *((float*)(freqs_sin->data)+i*head_width+j) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.rope_sin->data)+i*head_width+j));
            }
        }
        struct ggml_tensor * Q_without_cls_token_rot = ggml_rotate_half(ctx0, Q_without_cls_token);
        struct ggml_tensor * Q_rope_cos = ggml_mul(ctx0, Q_without_cls_token, ggml_repeat_3D(ctx0, freqs_cos, Q_without_cls_token));
        struct ggml_tensor * Q_rope_sin = ggml_mul(ctx0, Q_without_cls_token_rot, ggml_repeat_3D(ctx0, freqs_sin, Q_without_cls_token));
        struct ggml_tensor * Q_rope = ggml_cat(ctx0, Q_cls_token, ggml_add(ctx0, Q_rope_cos, Q_rope_sin));
        struct ggml_tensor * K_without_cls_token_rot = ggml_rotate_half(ctx0, K_without_cls_token);
        struct ggml_tensor * K_rope_cos = ggml_mul(ctx0, K_without_cls_token, ggml_repeat_3D(ctx0, freqs_cos, K_without_cls_token));
        struct ggml_tensor * K_rope_sin = ggml_mul(ctx0, K_without_cls_token_rot, ggml_repeat_3D(ctx0, freqs_sin, K_without_cls_token));
        struct ggml_tensor * K_rope = ggml_cat(ctx0, K_cls_token, ggml_add(ctx0, K_rope_cos, K_rope_sin));

        //attention
        struct ggml_tensor * scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        //*((float*)(scale->data)) = ggml_fp16_to_fp32(*((uint16_t*)(model.logit_scale->data)));
        *((float*)(scale->data)) = 0.125;
        Q = ggml_scale(ctx0, Q_rope, scale);
        // query @ key.transpose(-2, -1)
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K_rope, Q);
        // KQ = KQ.softmax(-1)
        struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ); //inplace
        // KQV = attn @ value
        struct ggml_tensor * V_trans = ggml_cpy(ctx0, ggml_transpose(ctx0, V), ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, patch_num*patch_num+1, head_width, n_head));
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);
        // KQV_merged = KQV.permute(0, 2, 1, 3)
        struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        // cur = KQV_merged.contiguous().view(n_embd, N)
        cur = ggml_cpy(ctx0,
                KQV_merged,
                ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, vision_width, patch_num*patch_num+1));

        //inner_attn_ln
        cur = ggml_norm(ctx0, cur);

        struct ggml_tensor * inner_attn_norm_weight = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
        for (int i = 0; i < vision_width; i++){
            *((float*)(inner_attn_norm_weight->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].inner_attn_norm_weight->data)+i));
        }
        struct ggml_tensor * inner_attn_norm_weight_broadcast = ggml_repeat(ctx0, inner_attn_norm_weight, cur);
        cur = ggml_mul(ctx0, cur, inner_attn_norm_weight_broadcast);

        struct ggml_tensor * inner_attn_norm_bias = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
        for (int i = 0; i < vision_width; i++){
            *((float*)(inner_attn_norm_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].inner_attn_norm_bias->data)+i));
        }
        struct ggml_tensor * inner_attn_norm_bias_broadcast = ggml_repeat(ctx0, inner_attn_norm_bias, cur);
        cur = ggml_add(ctx0, cur, inner_attn_norm_bias_broadcast);

        //projection
        struct ggml_tensor * bo = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
        for (int i = 0; i < vision_width; i++){
            *((float*)(bo->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].bo->data)+i));
        }
        struct ggml_tensor * bo_broadcast = ggml_repeat(ctx0, bo, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.vision_model.layers[il].wo, cur), bo_broadcast);
        

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);


        // feed-forward network
        
        //norm2
        cur = ggml_norm(ctx0, inpFF);

        struct ggml_tensor * norm2_weight = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
        for (int i = 0; i < vision_width; i++){
            *((float*)(norm2_weight->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].norm2_weight->data)+i));
        }
        struct ggml_tensor * norm2_weight_broadcast = ggml_repeat(ctx0, norm2_weight, cur);
        cur = ggml_mul(ctx0, cur, norm2_weight_broadcast);

        struct ggml_tensor * norm2_bias = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
        for (int i = 0; i < vision_width; i++){
            *((float*)(norm2_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].norm2_bias->data)+i));
        }
        struct ggml_tensor * norm2_bias_broadcast = ggml_repeat(ctx0, norm2_bias, cur);
        cur = ggml_add(ctx0, cur, norm2_bias_broadcast);

        //swiglu ffn
        struct ggml_tensor * b1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, ratio_width);
        struct ggml_tensor * b2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, ratio_width);
        struct ggml_tensor * b3 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
        for (int i = 0; i < ratio_width; i++){
            *((float*)(b1->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].b1->data)+i));
            *((float*)(b2->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].b2->data)+i));
        }
        for (int i = 0; i < vision_width; i++){
            *((float*)(b3->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].b3->data)+i));
        }
        struct ggml_tensor * broadcast_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ratio_width, patch_num*patch_num+1);
        struct ggml_tensor * b1_broadcast = ggml_repeat(ctx0, b1, broadcast_template);
        struct ggml_tensor * b2_broadcast = ggml_repeat(ctx0, b2, broadcast_template);
        struct ggml_tensor * b3_broadcast = ggml_repeat(ctx0, b3, cur);

        struct ggml_tensor * x1 = ggml_add(ctx0, ggml_mul_mat(ctx0,model.vision_model.layers[il].w1,cur), b1_broadcast);
        struct ggml_tensor * x2 = ggml_add(ctx0, ggml_mul_mat(ctx0,model.vision_model.layers[il].w2,cur), b2_broadcast);
        // SILU activation
        struct ggml_tensor * x1_silu = ggml_silu(ctx0, x1);
        cur = ggml_mul(ctx0, x1_silu, x2);

        //ffn_ln
        cur = ggml_norm(ctx0, cur);

        struct ggml_tensor * ffn_norm_weight = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, ratio_width);
        for (int i = 0; i < ratio_width; i++){
            *((float*)(ffn_norm_weight->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].ffn_norm_weight->data)+i));
        }
        struct ggml_tensor * ffn_norm_weight_broadcast = ggml_repeat(ctx0, ffn_norm_weight, cur);
        cur = ggml_mul(ctx0, cur, ffn_norm_weight_broadcast);

        struct ggml_tensor * ffn_norm_bias = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, ratio_width);
        for (int i = 0; i < ratio_width; i++){
            *((float*)(ffn_norm_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.layers[il].ffn_norm_bias->data)+i));
        }
        struct ggml_tensor * ffn_norm_bias_broadcast = ggml_repeat(ctx0, ffn_norm_bias, cur);
        cur = ggml_add(ctx0, cur, ffn_norm_bias_broadcast);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0,model.vision_model.layers[il].w3,cur), b3_broadcast);
        

        cur  = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }
    //norm
    inpL = ggml_norm(ctx0, inpL);

    struct ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
    for (int i = 0; i < vision_width; i++){
        *((float*)(norm_weight->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.norm_weight->data)+i));
    }
    struct ggml_tensor * norm_weight_broadcast = ggml_repeat(ctx0, norm_weight, inpL);
    inpL = ggml_mul(ctx0, inpL, norm_weight_broadcast);

    struct ggml_tensor * norm_bias = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, vision_width);
    for (int i = 0; i < vision_width; i++){
        *((float*)(norm_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.norm_bias->data)+i));
    }
    struct ggml_tensor * norm_bias_broadcast = ggml_repeat(ctx0, norm_bias, inpL);
    inpL = ggml_add(ctx0, inpL, norm_bias_broadcast);

    struct ggml_tensor * zero = ggml_new_i32(ctx0, 0);
    inpL = ggml_get_rows(ctx0, inpL, zero);

    //head
    struct ggml_tensor * bh = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_embd);
    for (int i = 0; i < n_embd; i++){
        *((float*)(bh->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.vision_model.head_bias->data)+i));
    }
    inpL = ggml_add(ctx0, ggml_mul_mat(ctx0, model.vision_model.head_weight, inpL), bh);

    ggml_build_forward_expand(&vision_gf, inpL);
    ggml_graph_compute       (ctx0, &vision_gf);

    //auto & image_features = lctx.image_features;
    //image_features.resize(n_embd);
    //memcpy(image_features.data(), (float *) ggml_get_data(inpL), sizeof(float)*n_embd);
    memcpy((float*)(image_features->data), (float *) ggml_get_data(inpL), sizeof(float)*n_embd);

    ggml_free(ctx0);
    }

    const int context_length = text_hparams.context_length; //77
    const int vocab_size = text_hparams.vocab_size; //49408
    const int text_width = text_hparams.width; //512
    const int heads = text_hparams.heads; //8
    const int text_head_width = text_width/heads; //64
    const int n_text_layer = text_hparams.layers; //12
    //const bool xattn = text_hparams.xattn;
    //const bool fusedLN = text_hparams.fusedLN;
    struct ggml_tensor * embds = ectx.text; //[3,77]
    const int total_label_num = embds->ne[1];
    struct ggml_tensor * text_features = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, total_label_num);

    //text tower
    {
    static size_t text_buf_size = 512u*1024*1024;
    static void * text_buf = malloc(text_buf_size);

    struct ggml_init_params text_params = {
        /*.mem_size   =*/ text_buf_size,
        /*.mem_buffer =*/ text_buf,
    };

    ggml_cgraph text_gf = {};
    text_gf.n_threads = n_threads;

    int eot_id = 0;
    for (int label_num = 0; label_num < total_label_num; label_num++){
        struct ggml_context * ctx1 = ggml_init(text_params);
        struct ggml_tensor * embd = ggml_new_tensor_1d(ctx1, GGML_TYPE_I32, context_length);
        memcpy(ggml_get_data(embd), (int *) ggml_get_data(embds) + label_num * context_length, sizeof(int)*context_length);
        //get eot_id
        for (int i = 0; i < context_length; i++){
            if (*((int*)(embd->data)+i) == vocab_size-1){
                eot_id = i;
                break;
            }
        }
        struct ggml_tensor * eot_id_tensor = ggml_new_i32(ctx1, eot_id);

        //Embedding
        struct ggml_tensor * inpL = ggml_get_rows(ctx1, model.text_model.token_embed, embd); //[77,512]

        //Positional Encoding
        struct ggml_tensor * positional_embedding = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, text_width, context_length);
        for (int i = 0; i < text_width*context_length; i++){
            *((float*)(positional_embedding->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.pos_embeddings->data)+i));
        }
        inpL = ggml_add(ctx1, inpL, positional_embedding);

        //Transformer layers
        for (int il = 0; il < n_text_layer; il++){
            struct ggml_tensor * inpSA = inpL;

            struct ggml_tensor * cur;

            // norm1
            cur = ggml_norm(ctx1, inpL);

            struct ggml_tensor * norm1_weight = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, text_width);
            for (int i = 0; i < text_width; i++){
                *((float*)(norm1_weight->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.layers[il].norm1_weight->data)+i));
            }
            struct ggml_tensor * norm1_weight_broadcast = ggml_repeat(ctx1, norm1_weight, cur);
            cur = ggml_mul(ctx1, cur, norm1_weight_broadcast);

            struct ggml_tensor * norm1_bias = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, text_width);
            for (int i = 0; i < text_width; i++){
                *((float*)(norm1_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.layers[il].norm1_bias->data)+i));
            }
            struct ggml_tensor * norm1_bias_broadcast = ggml_repeat(ctx1, norm1_bias, cur);
            cur = ggml_add(ctx1, cur, norm1_bias_broadcast);

            // in_projection
            struct ggml_tensor * in_proj_bias = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, text_width*3);
            for (int i = 0; i < text_width*3; i++){
                *((float*)(in_proj_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.layers[il].in_proj_bias->data)+i));
            }
            struct ggml_tensor * in_proj_bias_broadcast = ggml_repeat(ctx1, in_proj_bias, ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, text_width*3, context_length));

            struct ggml_tensor * q_k_v = ggml_add(ctx1, ggml_mul_mat(ctx1, model.text_model.layers[il].in_proj_weight, cur), in_proj_bias_broadcast); //[77,1536]

            struct ggml_tensor * Qcur = ggml_split_get_first(ctx1, q_k_v, 0, text_width);
            struct ggml_tensor * k_v = ggml_split_get_second(ctx1, q_k_v, 0, text_width);
            struct ggml_tensor * Kcur = ggml_split_get_first(ctx1, k_v, 0, text_width);
            struct ggml_tensor * Vcur = ggml_split_get_second(ctx1, k_v, 0, text_width); //[77,512]

            // Q = Qcur.contiguous().view(context_length=77, heads = 8, head_width=64).permute(0, 2, 1, 3).contiguous()
            struct ggml_tensor * Q =
                ggml_cpy(ctx1,
                ggml_permute(ctx1,
                            ggml_cpy(ctx1,
                                Qcur,
                                ggml_new_tensor_3d(ctx1, GGML_TYPE_F32, text_head_width, heads, context_length)),
                        0, 2, 1, 3),
                ggml_new_tensor_3d(ctx1, GGML_TYPE_F32, text_head_width, context_length, heads)); //[8,77,64]
            // K = Kcur.contiguous().view(context_length=77, heads = 8, head_width=64).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx1,
                            ggml_cpy(ctx1,
                                Kcur,
                                ggml_new_tensor_3d(ctx1, GGML_TYPE_F32, text_head_width, heads, context_length)),
                        0, 2, 1, 3); //[8,77,64]
            // V = Vcur.contiguous().view(context_length=77, heads = 8, head_width=64).permute(0, 2, 1, 3)
            struct ggml_tensor * V =
                ggml_cpy(ctx1,
                ggml_permute(ctx1,
                            ggml_cpy(ctx1,
                                Vcur,
                                ggml_new_tensor_3d(ctx1, GGML_TYPE_F32, text_head_width, heads, context_length)),
                        0, 2, 1, 3),
                ggml_new_tensor_3d(ctx1, GGML_TYPE_F32, text_head_width, context_length, heads)); //[8,77,64]

            //Attention
            struct ggml_tensor * scale = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, 1);
            //*((float*)(scale->data)) = ggml_fp16_to_fp32(*((uint16_t*)(model.logit_scale->data)));
            *((float*)(scale->data)) = 0.125;
            struct ggml_tensor * Q_scaled = ggml_scale(ctx1, Q, scale);
            // query @ key.transpose(-2, -1)
            struct ggml_tensor * KQ = ggml_mul_mat(ctx1, K, Q_scaled); // [8,77,77]
            // add attn bias
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx1, KQ, 0); //inplace
            // KQ = KQ.softmax(-1)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx1, KQ_masked); //inplace
            // KQV = attn @ value
            struct ggml_tensor * V_trans = ggml_cpy(ctx1, ggml_transpose(ctx1, V), ggml_new_tensor_3d(ctx1, GGML_TYPE_F32, context_length, text_head_width, heads));
            struct ggml_tensor * KQV = ggml_mul_mat(ctx1, V_trans, KQ_soft_max); //[8,77,64]
            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx1, KQV, 0, 2, 1, 3); //[77,8,64]
            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx1,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, text_width, context_length)); //[77,512]           

            //out_projection
            struct ggml_tensor * out_proj_bias = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, text_width);
            for (int i = 0; i < text_width; i++){
                *((float*)(out_proj_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.layers[il].out_proj_bias->data)+i));
            }
            struct ggml_tensor * out_proj_bias_broadcast = ggml_repeat(ctx1, out_proj_bias, cur);
            cur = ggml_add(ctx1, ggml_mul_mat(ctx1, model.text_model.layers[il].out_proj_weight, cur), out_proj_bias_broadcast);
        

            struct ggml_tensor * inpFF = ggml_add(ctx1, cur, inpSA); //[77,512]
    

            // feed-forward network
        

            //norm2
            cur = ggml_norm(ctx1, inpFF);

            struct ggml_tensor * norm2_weight = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, text_width);
            for (int i = 0; i < text_width; i++){
                *((float*)(norm2_weight->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.layers[il].norm2_weight->data)+i));
            }
            struct ggml_tensor * norm2_weight_broadcast = ggml_repeat(ctx1, norm2_weight, cur);
            cur = ggml_mul(ctx1, cur, norm2_weight_broadcast);

            struct ggml_tensor * norm2_bias = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, text_width);
            for (int i = 0; i < text_width; i++){
                *((float*)(norm2_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.layers[il].norm2_bias->data)+i));
            }
            struct ggml_tensor * norm2_bias_broadcast = ggml_repeat(ctx1, norm2_bias, cur);
            cur = ggml_add(ctx1, cur, norm2_bias_broadcast);

            //mlp
            //c_fc:Linear(in_features=512, out_features=2048, bias=True)
            struct ggml_tensor * c_fc_bias = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, text_width*4);
            for (int i = 0; i < text_width*4; i++){
                *((float*)(c_fc_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.layers[il].c_fc_bias->data)+i));
            }
            struct ggml_tensor * c_fc_bias_broadcast = ggml_repeat(ctx1, c_fc_bias, ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, text_width*4, context_length));
            cur = ggml_add(ctx1, ggml_mul_mat(ctx1, model.text_model.layers[il].c_fc_weight, cur), c_fc_bias_broadcast);
            //gelu:GELU(approximate=none)
            cur = ggml_gelu(ctx1, cur); //[77,2048]
            //c_proj:Linear(in_features=2048, out_features=512, bias=True)
            struct ggml_tensor * c_proj_bias = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, text_width);
            for (int i = 0; i < text_width; i++){
                *((float*)(c_proj_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.layers[il].c_proj_bias->data)+i));
            }
            struct ggml_tensor * c_proj_bias_broadcast = ggml_repeat(ctx1, c_proj_bias, ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, text_width, context_length));
            cur = ggml_add(ctx1, ggml_mul_mat(ctx1, model.text_model.layers[il].c_proj_weight, cur), c_proj_bias_broadcast);

            cur  = ggml_add(ctx1, cur, inpFF);

            // input for next layer
            inpL = cur; //[77,512]
        }

        //norm
        inpL = ggml_norm(ctx1, inpL);

        struct ggml_tensor * final_norm_weight = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, text_width);
        for (int i = 0; i < text_width; i++){
            *((float*)(final_norm_weight->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.ln_final_weight->data)+i));
        }
        struct ggml_tensor * norm_weight_broadcast = ggml_repeat(ctx1, final_norm_weight, inpL);
        inpL = ggml_mul(ctx1, inpL, norm_weight_broadcast);

        struct ggml_tensor * final_norm_bias = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, text_width);
        for (int i = 0; i < text_width; i++){
            *((float*)(final_norm_bias->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.ln_final_bias->data)+i));
        }
        struct ggml_tensor * final_norm_bias_broadcast = ggml_repeat(ctx1, final_norm_bias, inpL);
        inpL = ggml_add(ctx1, inpL, final_norm_bias_broadcast);


        struct ggml_tensor * text_proj = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, text_width, text_width);
        for (int i = 0; i < text_width*text_width; i++){
            *((float*)(text_proj->data)+i) = ggml_fp16_to_fp32(*((uint16_t*)(model.text_model.text_proj->data)+i));
        }
        struct ggml_tensor * text_proj_trans = ggml_cpy(ctx1, ggml_transpose(ctx1, text_proj), ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, text_width, text_width));
        //take features from the eot embedding (eot_token is the highest number in each sequence)
        //eot_token @ text_projection
        inpL = ggml_mul_mat(ctx1, text_proj_trans, ggml_get_rows(ctx1, inpL, eot_id_tensor));
        ggml_build_forward_expand(&text_gf, inpL);
        ggml_graph_compute       (ctx1, &text_gf);

        //memcpy(text_features.data() + label_num*text_width, (float *) ggml_get_data(inpL), sizeof(float)*text_width);
        memcpy((float *) ggml_get_data(text_features) + label_num*text_width, (float *) ggml_get_data(inpL), sizeof(float)*n_embd);

        ggml_free(ctx1);
    }
    }
    //nomalize image_features and text_features
    //image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = ggml_l2norm(ctx, image_features);
    //text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = ggml_l2norm(ctx, text_features);
    //text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    struct ggml_tensor * text_probs = ggml_soft_max(ctx, 
                                            ggml_scale(ctx, 
                                                ggml_mul_mat(ctx, text_features, image_features), 
                                                    ggml_new_f32(ctx, 100.0)));
    ggml_build_forward_expand(&result_gf, text_probs);
    ggml_graph_compute       (ctx, &result_gf);

    auto & label_probs = ectx.label_probs;
    label_probs.resize(total_label_num);
    memcpy(label_probs.data(), (float *) ggml_get_data(text_probs), sizeof(float)*total_label_num);

    ggml_free(ctx);

    fprintf(stderr, "\n");
    fprintf(stderr, "probs:\n");
    for (auto & label_prob : label_probs) {
        fprintf(stderr, "%f\n", label_prob);
    }

    ectx.t_eval_us += ggml_time_us() - t_start_us;

    return true;
}

//
// interface implementation
//

struct eva_context * eva_init_from_file(const char * path_model, const char * image_path, const char * text) {
    ggml_time_init();

    eva_context * ctx = new eva_context;

    //ggml_type type_memory = GGML_TYPE_F32;

    if (!eva_model_load(path_model, image_path, text, *ctx)) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

int eva_eval(
        struct eva_context * ctx, int n_threads) {
    if (!eva_eval_internal(*ctx, n_threads)) {
        fprintf(stderr, "%s: failed to eval\n", __func__);
        return 1;
    }

    return 0;
}

void eva_print_timings(struct eva_context * ctx) {
    const int64_t t_end_us = ggml_time_us();

    fprintf(stderr, "\n");
    fprintf(stderr, "%s:     load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0f);
    fprintf(stderr, "%s:     eval time = %8.2f ms\n", __func__, 1e-3f * ctx->t_eval_us);
    fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us)/1000.0f);
}

const char * eva_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";

    return s.c_str();
}