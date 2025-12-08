#pragma once

#include "./ggml.h"
#include "./ggml-backend.h"
#include "ggml-cuda/common.cuh"
#include "TaskScheduler/Public/TaskScheduler.h"

#ifdef  __cplusplus
extern "C" {
#endif

	
	typedef bool (*ggml_cuda_compute_forward_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
	typedef enum ggml_status (*ggml_backend_cuda_graph_compute_t)(ggml_backend_t backend, ggml_cgraph * cgraph);

	// backend API
	GGML_API ggml_backend_t ggml_backend_cuda_init(int device);

	GGML_API bool ggml_backend_is_cuda(ggml_backend_t backend);

	// device buffer
	GGML_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

	// split tensor buffer that splits matrices by rows across multiple devices
	GGML_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

	// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
	GGML_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

	GGML_API int  ggml_backend_cuda_get_device_count(void);
	GGML_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
	GGML_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

	GGML_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
	GGML_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

	GGML_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

	GGML_API bool ggml_backend_buft_is_cuda_split(ggml_backend_buffer_type_t buft);
	GGML_API void ggml_cuda_set_peer_access(const int n_tokens, int main_device);

	GGML_API void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
	GGML_API void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

	GGML_API void register_ggml_backend_cuda_graph_compute(ggml_backend_cuda_graph_compute_t func);
	GGML_API enum ggml_status ggml_backend_cuda_graph_compute_scheduled(ggml_backend_t backend, ggml_cgraph * cgraph);
	GGML_API enum ggml_status ggml_backend_cuda_graph_compute_default(ggml_backend_t backend, ggml_cgraph * cgraph);
	
#ifdef __cplusplus
}
#endif