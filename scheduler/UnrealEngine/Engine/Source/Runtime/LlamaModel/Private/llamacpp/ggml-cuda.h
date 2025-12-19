#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "TaskScheduler/Public/TaskScheduler.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16
	typedef void (*ggml_cuda_func_t)(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
    typedef bool (*ggml_cuda_compute_forward_t)(struct ggml_compute_params *params, struct ggml_tensor *tensor);

// Always success. To check if CUDA is actually loaded, use `ggml_cublas_loaded`.
    GGML_API GGML_CALL void   ggml_init_cublas(void);

// Returns `true` if there are available CUDA devices and cublas loads successfully; otherwise, it returns `false`.
    GGML_API GGML_CALL bool   ggml_cublas_loaded(void);

    GGML_API GGML_CALL void * ggml_cuda_host_malloc(size_t size);
    GGML_API GGML_CALL void   ggml_cuda_host_free(void * ptr);

    GGML_API GGML_CALL bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);

    GGML_API GGML_CALL int    ggml_cuda_get_device_count(void);
    GGML_API GGML_CALL void   ggml_cuda_get_device_description(int device, char * description, size_t description_size);

// backend API
	GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device);

	GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

	GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);
// split tensor buffer that splits matrices by rows across multiple devices
	GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
	GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

	GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
	GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
	GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
	
	GGML_API GGML_CALL void ggml_cuda_set_peer_access(const int n_tokens);
	GGML_API GGML_CALL void ggml_cuda_repeat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_get_rows(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_add(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_acc(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_div(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_gelu(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_silu(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_gelu_quick(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_tanh(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_relu(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_hardsigmoid(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_hardswish(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_leaky_relu(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_sqr(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_norm(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_group_norm(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_concat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_upscale(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_pad(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_rms_norm(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_dup(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_diag_mask_inf(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_soft_max(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_rope(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_alibi(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_pool2d(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_im2col(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_sum_rows(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_argsort(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_nop(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_mul_mat_id(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_scale(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_clamp(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	GGML_API GGML_CALL void ggml_cuda_cpy(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
    GGML_API GGML_CALL void register_ggml_cuda_compute_forward(ggml_cuda_compute_forward_t func);
	GGML_CALL bool ggml_cuda_compute_forward_scheduled(struct ggml_compute_params * params, struct ggml_tensor * tensor);

#ifdef __cplusplus
	}
#endif