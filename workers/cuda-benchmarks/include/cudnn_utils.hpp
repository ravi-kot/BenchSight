#pragma once

#ifndef BENCHSIGHT_HAVE_CUDNN
#define BENCHSIGHT_HAVE_CUDNN 0
#endif

#if BENCHSIGHT_HAVE_CUDNN

#include <cudnn.h>

#include <stdexcept>
#include <string>

inline void cudnn_check(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(status));
    }
}

#endif
