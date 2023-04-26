// Various helper functions and utilities

#pragma once

#include <string>
#include <vector>
#include <random>
#include <thread>

//
// CLIP argument parsing
//

struct eva_params {
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());

    std::string model  = "models/EVA02-CLIP-B-16/ggml-model-f16.bin"; // model path
    std::string img    = ""; // image path
    std::string text   = ""; // text
};

bool params_parse(int argc, char ** argv, eva_params & params);

void print_usage(int argc, char ** argv, const eva_params & params);
