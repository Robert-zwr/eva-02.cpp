#include "utils.h"
#include "ggml.h"
#include "eva.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>


int main(int argc, char ** argv) {
    ggml_time_init();
    eva_params params;
    params.model = "./models/EVA02-CLIP-B-16/ggml-model-f16.bin";
    //params.img = "./temp/image.bin";
    params.img = "CLIP.png";
    params.text = "a diagram/a dog/a cat";

    if (params_parse(argc, argv, params) == false) {
        return 1;
    }

    eva_context * ctx;

    // load the model
    {
        ctx = eva_init_from_file(params.model.c_str(), params.img.c_str(), params.text.c_str());

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), eva_print_system_info());
    }

    // inference
    if (eva_eval(ctx, params.n_threads)) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return 1;
    }

    eva_print_timings(ctx);
    return 0;
}