#ifndef EVA_H
#define EVA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef EVA_SHARED
#    ifdef _WIN32
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define EVA_API
#endif

#define EVA_FILE_VERSION 1
#define EVA_FILE_MAGIC 0x67676d66 // 'ggmf' in hex
#define EVA_FILE_MAGIC_UNVERSIONED 0x67676d6c // pre-versioned files

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct eva_context;

    EVA_API struct eva_context * eva_init_from_file(
                             const char * path_model);

#ifdef __cplusplus
}
#endif

#endif
