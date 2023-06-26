#ifndef IMAGE_TYPES_H
#define IMAGE_TYPES_H

#include "core.h"

#include <stdio.h>

struct image_t;

#define MAX_FILE_EXTENSIONS 16
#define MAX_SIGNATURES      16

typedef struct image_type_t {

    const char *file_type;
    const char *file_extensions[MAX_FILE_EXTENSIONS];
    const char *signatures[MAX_SIGNATURES];

    int (*call_back)(image_t*, FILE*);

} image_type_t;

extern const image_type_t image_types[];
extern const size_t num_image_types;

#endif
