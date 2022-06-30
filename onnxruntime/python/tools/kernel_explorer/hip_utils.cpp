#include "hip_utils.h"

const char* rocblas_get_error_name(rocblas_status status) {
  switch (status) {
    case rocblas_status_success:
      return "rocblas_status_success";
    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";
    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";
    case rocblas_status_perf_degraded:
      return "rocblas_status_perf_degraded";
    case rocblas_status_size_query_mismatch:
      return "rocblas_status_size_query_mismatch";
    case rocblas_status_size_increased:
      return "rocblas_status_size_increased";
    case rocblas_status_size_unchanged:
      return "rocblas_status_size_unchanged";
    case rocblas_status_invalid_value:
      return "rocblas_status_invalid_value";
    case rocblas_status_continue:
      return "rocblas_status_continue";
    case rocblas_status_check_numerics_fail:
      return "rocblas_status_check_numerics_fail";
    default:
      return "rocblas_unknown_error";
  }
}
