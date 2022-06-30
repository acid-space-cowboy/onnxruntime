#pragma once

#include <cstdio>
#include <hip/hip_runtime.h>

#include <rocblas.h>

#define HIP_CHECKED_CALL(expr)                                                             \
  do {                                                                                     \
    auto status = expr;                                                                    \
    if (status != hipSuccess) {                                                            \
      std::printf("HIP Error at %s:%d\n    Error name  : %s\n    Error string: %s\n",      \
                  __FILE__, __LINE__, hipGetErrorName(status), hipGetErrorString(status)); \
      std::abort();                                                                        \
    }                                                                                      \
  } while (0)

const char* rocblas_get_error_name(rocblas_status status);

#define ROCBLAS_CHECKED_CALL(expr)                                     \
  do {                                                                 \
    auto status = expr;                                                \
    if (status != rocblas_status_success) {                            \
      std::printf("rocBLAS Error at %s:%d\n    Error name  : %s\n",    \
                  __FILE__, __LINE__, rocblas_get_error_name(status)); \
      std::abort();                                                    \
    }                                                                  \
  } while (0)
