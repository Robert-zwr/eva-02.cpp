#include "eva.h"

#include "ggml.h"

#include <cinttypes>
#include <fstream>
#include <random>
#include <unordered_map>
#include <queue>
#include <regex>
#include <cassert>
#include <cstring>


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

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
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

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;
};

//
// model loading
//

static bool eva_model_load(
            const std::string & fname,
            eva_context & lctx)
    {
    fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    const int64_t t_start_us = ggml_time_us();

    lctx.t_start_us = t_start_us;

    std::vector<char> f_buf(1024*1024);

    auto & model = lctx.model;
    auto & vocab = lctx.vocab;

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

    int n_ff = 0;

    // load hparams
    {
        auto & hparams = model.hparams;
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
        const int n_head   = hparams.text_hparams.heads;  //8
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
        model.vision_model.pos_embeddings = ggml_new_tensor_2d(ctx, vtype, patch_num*patch_num+1, width);
        model.vision_model.patch_embed_weight = ggml_new_tensor_4d(ctx, vtype, width, 3, patch_size, patch_size);
        model.vision_model.patch_embed_bias = ggml_new_tensor_1d(ctx, vtype, width);
        model.vision_model.rope_cos = ggml_new_tensor_2d(ctx, vtype, patch_num*patch_num, head_width);
        model.vision_model.rope_sin = ggml_new_tensor_2d(ctx, vtype, patch_num*patch_num, head_width);


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

            layer.rope_cos = ggml_new_tensor_2d(ctx, wtype, patch_num*patch_num, head_width);
            layer.rope_sin = ggml_new_tensor_2d(ctx, wtype, patch_num*patch_num, head_width);

            layer.norm2_weight = ggml_new_tensor_1d(ctx, wtype, width);
            layer.norm2_bias   = ggml_new_tensor_1d(ctx, wtype, width);

            layer.w1 = ggml_new_tensor_2d(ctx, wtype, width*mlp_ratio, width);
            layer.b1 = ggml_new_tensor_1d(ctx, wtype, width*mlp_ratio);
            layer.w2 = ggml_new_tensor_2d(ctx, wtype, width*mlp_ratio, width);
            layer.b2 = ggml_new_tensor_1d(ctx, wtype, width*mlp_ratio);
            layer.ffn_norm_weight = ggml_new_tensor_1d(ctx, wtype, width*mlp_ratio);
            layer.ffn_norm_bias   = ggml_new_tensor_1d(ctx, wtype, width*mlp_ratio);
            layer.w3 = ggml_new_tensor_2d(ctx, wtype, width, width*mlp_ratio);
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
        model.vision_model.head_weight = ggml_new_tensor_2d(ctx, vtype, n_embd, width);
        model.vision_model.head_bias   = ggml_new_tensor_1d(ctx, vtype, n_embd);

        model.tensors["visual.norm.weight"] = model.vision_model.norm_weight;
        model.tensors["visual.norm.bias"] = model.vision_model.norm_bias;
        model.tensors["visual.head.weight"] = model.vision_model.head_weight;
        model.tensors["visual.head.bias"] = model.vision_model.head_bias;


        //text
        const int context_length   = hparams.text_hparams.context_length;  //77
        const int vocab_size   = hparams.text_hparams.vocab_size;  //49408
        const int text_width   = hparams.text_hparams.width;  //512
        const int n_head   = hparams.text_hparams.heads;  //8
        const int n_layer   = hparams.text_hparams.layers;  //12

        model.text_model.layers.resize(n_layer);

        model.text_model.pos_embeddings = ggml_new_tensor_2d(ctx, wtype, context_length, text_width);
        model.text_model.text_proj = ggml_new_tensor_2d(ctx, wtype, text_width, text_width);
        model.text_model.token_embed = ggml_new_tensor_2d(ctx, wtype, vocab_size, text_width);

        model.tensors["text.positional_embedding"] = model.text_model.pos_embeddings;
        model.tensors["text.text_projection"] = model.text_model.text_proj;
        model.tensors["text.token_embedding.weight"] = model.text_model.token_embed;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.text_model.layers[i];

            layer.norm1_weight = ggml_new_tensor_1d(ctx, vtype, text_width);
            layer.norm1_bias   = ggml_new_tensor_1d(ctx, vtype, text_width);
            layer.in_proj_weight = ggml_new_tensor_2d(ctx, vtype, 3*text_width, text_width);
            layer.in_proj_bias = ggml_new_tensor_1d(ctx, vtype, 3*text_width);
            layer.out_proj_weight = ggml_new_tensor_2d(ctx, vtype, text_width, text_width);
            layer.out_proj_bias = ggml_new_tensor_1d(ctx, vtype, text_width);
            layer.norm2_weight = ggml_new_tensor_1d(ctx, vtype, text_width);
            layer.norm2_bias   = ggml_new_tensor_1d(ctx, vtype, text_width);
            layer.c_fc_weight = ggml_new_tensor_2d(ctx, vtype, 4*text_width, text_width);
            layer.c_fc_bias = ggml_new_tensor_1d(ctx, vtype, 4*text_width);
            layer.c_proj_weight = ggml_new_tensor_2d(ctx, vtype, text_width, 4*text_width);
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
            int32_t ftype; //1

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

    lctx.t_load_us = ggml_time_us() - t_start_us;

    return true;
}

//
// interface implementation
//

struct eva_context * eva_init_from_file(const char * path_model) {
    ggml_time_init();

    eva_context * ctx = new eva_context;

    ggml_type type_memory = GGML_TYPE_F32;

    if (!eva_model_load(path_model, *ctx)) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }

    return ctx;
}