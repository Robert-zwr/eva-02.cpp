#include "utils.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>

 #if defined(_MSC_VER) || defined(__MINGW32__)
 #include <malloc.h> // using malloc.h with MSC/MINGW
 #elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
 #include <alloca.h>
 #endif

bool params_parse(int argc, char ** argv, eva_params & params) {
    // determine sensible default number of threads.
    // std::thread::hardware_concurrency may not be equal to the number of cores, or may return 0.
    params.n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-i" || arg == "--image") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.img = argv[i];
        } else if (arg == "-c" || arg == "--caption") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.text = argv[i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv, params);
        exit(1);
    }

    return true;
}

void print_usage(int /*argc*/, char ** argv, const eva_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i FNAME, --image FNAME     path to a image to be evaluated\n");
    fprintf(stderr, "  -c 'str1/str2/str3', --caption 'str1/str2/str3'\n");
    fprintf(stderr, "                        optional captions/labels of the image,spilited by '/',surrounded by '' or \"\"\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "\n");
}
