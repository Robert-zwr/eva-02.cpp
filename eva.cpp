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

    struct ggml_tensor * image;

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
            const std::string & image_path,
            eva_context & lctx)
    {
    fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    const int64_t t_start_us = ggml_time_us();

    lctx.t_start_us = t_start_us;

    std::vector<char> f_buf(1024*1024);

    auto & model = lctx.model;
    auto & image = lctx.image;
    //auto & vocab = lctx.vocab;


    //load image
    auto fin = std::ifstream(image_path, std::ios::binary);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, image_path.c_str());
        return false;
    }
    int32_t dims;
    fin.read(reinterpret_cast<char *>(&dims), sizeof(dims));
    if (dims != 3) {
        fprintf(stderr, "%s: image tensor has wrong size in image file: got %d, expected %d\n",
                __func__, dims, 3);
        return false;
    }

    int32_t nelements = 1;
    int32_t ne[4] = {dims, 1, 1, 1};
    for (int i = 0; i < dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
    }
    static size_t image_buf_size = 1.2*ne[0]*ne[1]*ne[2]*ggml_type_sizef(GGML_TYPE_F32);
    static void * image_buf = malloc(image_buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/ image_buf_size,
        /*.mem_buffer =*/ image_buf,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    image = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, ne[0], ne[1], ne[2]);
    fin.read(reinterpret_cast<char *>(image->data), ggml_nbytes(image));
    fin.close();


    fin = std::ifstream(fname, std::ios::binary);
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

    lctx.t_load_us = ggml_time_us() - t_start_us;

    return true;
}

typedef unsigned short ushort;//占用2个字节
typedef unsigned int uint;    //占用4个字节
 
uint as_uint(const float x) {
    return *(uint*)&x;
}
float as_float(const uint x) {
    return *(float*)&x;
}
 
float half_to_float(const ushort x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint e = (x&0x7C00)>>10; // exponent
    const uint m = (x&0x03FF)<<13; // mantissa
    const uint v = as_uint((float)m)>>23; // evil log2 bit hack to count leading zeros in denormalized format
    return as_float((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FE000))); // sign : normalized : denormalized
}

// evaluate the transformer
//
//   - lctx:      llama context
//   - tokens:    new batch of tokens to process
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//
static bool eva_eval_internal(
        eva_context & lctx,
        const int   n_threads) {
    const int64_t t_start_us = ggml_time_us();

    //const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & vision_hparams = hparams.vision_hparams;
    const auto & text_hparams = hparams.text_hparams;

    const int n_embd  = hparams.n_embd;
    const int image_size = vision_hparams.image_size;
    const int n_vision_layer = vision_hparams.layers;
    const int vision_width = vision_hparams.width;  //768
    const int head_width = vision_hparams.head_width;
    const int patch_size = vision_hparams.patch_size; //16
    const int patch_num   = image_size/patch_size;  //14

    const int context_length = text_hparams.context_length;
    const int vocab_size = text_hparams.vocab_size;
    const int text_width = text_hparams.width;
    const int heads = text_hparams.heads;
    const int n_text_layer = text_hparams.layers;
    const bool xattn = text_hparams.xattn;
    const bool fusedLN = text_hparams.fusedLN;
    //const int n_ctx   = hparams.n_ctx;
    //const int n_head  = hparams.n_head;
    //const int n_vocab = hparams.n_vocab;
    //const int n_rot   = hparams.n_embd/hparams.n_head;

    auto & mem_per_token = lctx.mem_per_token;

    // TODO: fix this hardcoded size
    static size_t buf_size = 512u*1024*1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
    };

    ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    std::string template_path = "/home/zwr/EVA_env/eva-02.cpp/temp/template.bin";
    std::vector<char> f_buf(1024*1024);
    auto fin = std::ifstream(template_path, std::ios::binary);
    fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, template_path.c_str());
        return false;
    }
    struct ggml_context * ctx1 = ggml_init(params);
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx1, GGML_TYPE_F16, 1, 3, 4);
    fin.read(reinterpret_cast<char *>(a->data), ggml_nbytes(a));
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, 4, 3);
    fin.read(reinterpret_cast<char *>(b->data), ggml_nbytes(b));
    fin.close();

    struct ggml_tensor * c = ggml_conv_1d_1s(ctx1, a, b);
    ggml_build_forward_expand(&gf, c);
    ggml_graph_compute       (ctx1, &gf);

    ggml_free(ctx1);

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_tensor * vision_proj = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, vision_width, patch_num*patch_num);
    uint16_t * conv_kernel = (uint16_t*)malloc(patch_size*patch_size*3*sizeof(uint16_t));
    uint16_t * patch = (uint16_t*)malloc(patch_size*patch_size*3*sizeof(uint16_t));
    const int n = ((patch_size*patch_size*3+31) & ~31);
    for (int i = 0; i < vision_width; i++) {
        memcpy(conv_kernel, (unsigned short*)(model.vision_model.patch_embed_weight->data)+i*3*patch_size*patch_size, patch_size*patch_size*3*sizeof(unsigned short));
        float bias = ggml_fp16_to_fp32(*((unsigned short*)(model.vision_model.patch_embed_bias->data)+i));
        for (int j = 0; j < patch_num*patch_num; j++){
            for (int c = 0; c < 3; c++){
                for (int k = 0; k < patch_size; k++){
                    for (int m = 0; m < patch_size; m++){
                        patch[c*patch_size*patch_size+k*patch_size+m] = ggml_fp32_to_fp16(*(((float*)(lctx.image->data))+(j/patch_num)*image_size*patch_size+(j%patch_num)*patch_size+c*image_size*image_size+k*image_size+m));
                    }
                }
            }
            float v = 0.0f;
            ggml_vector_dot_f16(n, &v, conv_kernel, patch);
            v += bias;
            memcpy((float*)(vision_proj->data)+j*vision_width+i, &v, sizeof(float));
        }
    }
    //half_to_float(*(unsigned short*)model.vision_model.cls_token->data)==0.12561
    //half_to_float(*(((unsigned short*)model.vision_model.cls_token->data)+1))==-0.595214844
    //*((float*)(lctx.image->data)+17070)==0.324509382([0, 76, 46])
    //half_to_float(*(unsigned short*)model.vision_model.patch_embed_weight->data)==-0.0361633301
/*
    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, tokens, N*ggml_element_size(embd));

    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.tok_embeddings, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // norm
        {
            cur = ggml_rms_norm(ctx0, inpL);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].attention_norm, cur),
                        cur);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_rope(ctx0,
                            ggml_cpy(ctx0,
                                Qcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                            n_past, n_rot, 0),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_rope(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            n_past, n_rot, 1),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V_trans =
                ggml_cpy(ctx0,
                    ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            1, 2, 0, 3),
                    ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_past + N, n_embd/n_head, n_head));

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
        }

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF);

                // cur = ffn_norm*cur
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ffn_norm, cur),
                        cur);
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);


            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);

            // SILU activation
            cur = ggml_silu(ctx0, cur);

            cur = ggml_mul(ctx0, cur, tmp);

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
        }

        cur  = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    // norm
    {
        inpL = ggml_rms_norm(ctx0, inpL);

        // inpL = norm*inpL
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.norm, inpL),
                    inpL);
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.output, inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    auto & logits_out = lctx.logits;

    if (lctx.logits_all) {
        logits_out.resize(n_vocab * N);
        memcpy(logits_out.data(), (float *) ggml_get_data(inpL), sizeof(float)*n_vocab*N);
    } else {
        // return result for just the last token
        logits_out.resize(n_vocab);
        memcpy(logits_out.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
    }

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //fprintf(stderr, "used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    // measure the performance only for the single-token evals
    if (N == 1) {
        lctx.t_eval_us += ggml_time_us() - t_start_us;
        lctx.n_eval++;
    }
*/
    return true;
}

//
// interface implementation
//

struct eva_context * eva_init_from_file(const char * path_model, const char * image_path) {
    ggml_time_init();

    eva_context * ctx = new eva_context;

    ggml_type type_memory = GGML_TYPE_F32;

    if (!eva_model_load(path_model, image_path, *ctx)) {
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