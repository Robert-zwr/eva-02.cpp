#include "utils.h"
#include "ggml.h"
#include "llama.h"
#include "eva.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

#if defined (_WIN32)
#pragma comment(lib,"kernel32.lib")
extern "C" __declspec(dllimport) void* __stdcall GetStdHandle(unsigned long nStdHandle);
extern "C" __declspec(dllimport) int __stdcall GetConsoleMode(void* hConsoleHandle, unsigned long* lpMode);
extern "C" __declspec(dllimport) int __stdcall SetConsoleMode(void* hConsoleHandle, unsigned long dwMode);
#endif


int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;
    params.model = "/home/zwr/EVA_env/eva-02.cpp/models/EVA02-CLIP-B-16/ggml-model-f16.bin";
    params.img = "/home/zwr/EVA_env/eva-02.cpp/temp/image.bin";

    eva_context * ctx;

    // load the model
    {
        ctx = eva_init_from_file(params.model.c_str(), params.img.c_str());

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    // inference
    if (eva_eval(ctx, params.n_threads)) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return 1;
    }

    fprintf(stderr, "done\n");
    return 0;
}