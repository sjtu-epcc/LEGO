#pragma once

#include "CoreMinimal.h"
#include "./ggml-cuda.h"
#include "./ggml.h"
#include "./ggml-impl.h"
#include "./ggml-backend-impl.h"
#include "../../../SchedulingSystem/TaskScheduler/Public/TaskScheduler.h"
#include "cuda_runtime.h"
#include <cinttypes>
#include <string>
#include <DistributedBuildInterface/Public/DistributedBuildControllerInterface.h>

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/rwkv-wkv.cuh"

DECLARE_STATS_GROUP(TEXT("Inference"), STATGROUP_LLAMA, STATCAT_Advanced);
DECLARE_CYCLE_STAT_EXTERN(TEXT("Inference Task time"), STAT_Inference, STATGROUP_LLAMA, );
DECLARE_CYCLE_STAT_EXTERN(TEXT("Inference Sync time"), STAT_InferenceSync, STATGROUP_LLAMA, );
DEFINE_STAT(STAT_Inference);
DEFINE_STAT(STAT_InferenceSync);

TAtomic<int> GGMLOperatorTaskCount = 0;

class FMonitorGGMLOperatorTask : public FNonAbandonableTask
{
	friend class FAutoDeleteAsyncTask<FMonitorGGMLOperatorTask>;

	cudaEvent_t Event;

public:
	FMonitorGGMLOperatorTask(cudaEvent_t Event): Event(Event) {}
	
	void DoWork()
	{
		SCOPE_CYCLE_COUNTER(STAT_InferenceSync)
		
		while (true) {
			cudaError_t status = cudaEventQuery(Event);
			if (status == cudaSuccess) {
				std::cout << "Kernel execution completed!" << std::endl;
				break;
			} else if (status != cudaErrorNotReady) {
				std::cerr << "Error occurred while querying event." << std::endl;
				break;
			}
		}
	}
	
	FORCEINLINE TStatId GetStatId() const
	{
		RETURN_QUICK_DECLARE_CYCLE_STAT(FMonitorGGMLOperatorTask, STATGROUP_ThreadPoolAsyncTasks);
	}
};

bool ggml_cuda_compute_forward_scheduled(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
	fprintf(stderr, "register ggml_cuda_compute_forward succeeded\n");
	
	cudaEvent_t Event;
	cudaEventCreate(&Event);
	if (dst->src[0] != nullptr && ggml_backend_buft_is_cuda_split(dst->src[0]->buffer->buft)) {
		ggml_cuda_set_peer_access(dst->src[1]->ne[1], ctx.device);
	}
	
	TArray<int> Input;
    FString Name;
	
	//FTaskDetector::GetInstance().SetIsInInferring(true);
	
	SCOPE_CYCLE_COUNTER(STAT_Inference);
	
	switch (dst->op)
	{
	case GGML_OP_ARGMAX:
		ggml_cuda_argmax(ctx, dst);
		break;
	case GGML_OP_COUNT_EQUAL:
		ggml_cuda_count_equal(ctx, dst);
		break;
	case GGML_OP_REPEAT:
		ggml_cuda_op_repeat(ctx, dst);
		break;
	case GGML_OP_REPEAT_BACK:
		ggml_cuda_op_repeat_back(ctx, dst);
		break;
	case GGML_OP_GET_ROWS:
		ggml_cuda_op_get_rows(ctx, dst);
		break;
	case GGML_OP_DUP:
		ggml_cuda_dup(ctx, dst);
		break;
	case GGML_OP_CPY:
		ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
		break;
	case GGML_OP_CONT:
		ggml_cuda_dup(ctx, dst);
		break;
	case GGML_OP_ADD:
	case GGML_OP_ADD1: // TODO: more efficient implementation
		ggml_cuda_op_add(ctx, dst);
		break;
	case GGML_OP_SUB:
		ggml_cuda_op_sub(ctx, dst);
		break;
	case GGML_OP_ACC:
		ggml_cuda_op_acc(ctx, dst);
		break;
	case GGML_OP_MUL:
		ggml_cuda_op_mul(ctx, dst);
		break;
	case GGML_OP_DIV:
		ggml_cuda_op_div(ctx, dst);
		break;
	case GGML_OP_UNARY:
		switch (ggml_get_unary_op(dst)) {
	case GGML_UNARY_OP_NEG:
		ggml_cuda_op_neg(ctx, dst);
			break;
	case GGML_UNARY_OP_STEP:
		ggml_cuda_op_step(ctx, dst);
			break;
	case GGML_UNARY_OP_GELU:
		ggml_cuda_op_gelu(ctx, dst);
			break;
	case GGML_UNARY_OP_SILU:
		ggml_cuda_op_silu(ctx, dst);
			break;
	case GGML_UNARY_OP_GELU_QUICK:
		ggml_cuda_op_gelu_quick(ctx, dst);
			break;
	case GGML_UNARY_OP_TANH:
		ggml_cuda_op_tanh(ctx, dst);
			break;
	case GGML_UNARY_OP_RELU:
		ggml_cuda_op_relu(ctx, dst);
			break;
	case GGML_UNARY_OP_SIGMOID:
		ggml_cuda_op_sigmoid(ctx, dst);
			break;
	case GGML_UNARY_OP_HARDSIGMOID:
		ggml_cuda_op_hardsigmoid(ctx, dst);
			break;
	case GGML_UNARY_OP_HARDSWISH:
		ggml_cuda_op_hardswish(ctx, dst);
			break;
	case GGML_UNARY_OP_EXP:
		ggml_cuda_op_exp(ctx, dst);
			break;
	default:
		return false;
		}
		break;
	case GGML_OP_NORM:
		ggml_cuda_op_norm(ctx, dst);
		break;
	case GGML_OP_GROUP_NORM:
		ggml_cuda_op_group_norm(ctx, dst);
		break;
	case GGML_OP_CONCAT:
		ggml_cuda_op_concat(ctx, dst);
		break;
	case GGML_OP_UPSCALE:
		ggml_cuda_op_upscale(ctx, dst);
		break;
	case GGML_OP_PAD:
		ggml_cuda_op_pad(ctx, dst);
		break;
	case GGML_OP_ARANGE:
		ggml_cuda_op_arange(ctx, dst);
		break;
	case GGML_OP_TIMESTEP_EMBEDDING:
		ggml_cuda_op_timestep_embedding(ctx, dst);
		break;
	case GGML_OP_LEAKY_RELU:
		ggml_cuda_op_leaky_relu(ctx, dst);
		break;
	case GGML_OP_RMS_NORM:
		ggml_cuda_op_rms_norm(ctx, dst);
		break;
	case GGML_OP_MUL_MAT:
		if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
			GGML_LOG_ERROR("%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, dst->name, dst->src[0]->ne[3], dst->src[1]->ne[3]);
			return false;
		} else {
			ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
		}
		break;
	case GGML_OP_MUL_MAT_ID:
		ggml_cuda_mul_mat_id(ctx, dst);
		break;
	case GGML_OP_OUT_PROD:
		ggml_cuda_out_prod(ctx, dst);
		break;
	case GGML_OP_SCALE:
		ggml_cuda_op_scale(ctx, dst);
		break;
	case GGML_OP_SQR:
		ggml_cuda_op_sqr(ctx, dst);
		break;
	case GGML_OP_SQRT:
		ggml_cuda_op_sqrt(ctx, dst);
		break;
	case GGML_OP_SIN:
		ggml_cuda_op_sin(ctx, dst);
		break;
	case GGML_OP_COS:
		ggml_cuda_op_cos(ctx, dst);
		break;
	case GGML_OP_CLAMP:
		ggml_cuda_op_clamp(ctx, dst);
		break;
	case GGML_OP_NONE:
	case GGML_OP_RESHAPE:
	case GGML_OP_VIEW:
	case GGML_OP_PERMUTE:
	case GGML_OP_TRANSPOSE:
			break;
	case GGML_OP_DIAG_MASK_INF:
		ggml_cuda_op_diag_mask_inf(ctx, dst);
		break;
	case GGML_OP_SOFT_MAX:
		ggml_cuda_op_soft_max(ctx, dst);
		break;
	case GGML_OP_ROPE:
		ggml_cuda_op_rope(ctx, dst);
		break;
	case GGML_OP_IM2COL:
		ggml_cuda_op_im2col(ctx, dst);
		break;
	case GGML_OP_CONV_TRANSPOSE_1D:
		ggml_cuda_op_conv_transpose_1d(ctx,dst);
		break;
	case GGML_OP_POOL_2D:
		ggml_cuda_op_pool2d(ctx, dst);
		break;
	case GGML_OP_SUM:
		ggml_cuda_op_sum(ctx, dst);
		break;
	case GGML_OP_SUM_ROWS:
		ggml_cuda_op_sum_rows(ctx, dst);
		break;
	case GGML_OP_ARGSORT:
		ggml_cuda_op_argsort(ctx, dst);
		break;
	case GGML_OP_FLASH_ATTN_EXT:
		ggml_cuda_flash_attn_ext(ctx, dst);
		break;
	case GGML_OP_CROSS_ENTROPY_LOSS:
		ggml_cuda_cross_entropy_loss(ctx, dst);
		break;
	case GGML_OP_RWKV_WKV:
		ggml_cuda_op_rwkv_wkv(ctx, dst);
		break;
	case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
		ggml_cuda_cross_entropy_loss_back(ctx, dst);
		break;
	case GGML_OP_OPT_STEP_ADAMW:
		ggml_cuda_opt_step_adamw(ctx, dst);
		break;
	default:
		return false;
	}
	
	//FTaskDetector::GetInstance().SetIsInInferring(false);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
		CUDA_CHECK(err);
	}
	
    return true;
}

bool ggml_cuda_compute_forward_default(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
	
	fprintf(stderr, "register ggml_cuda_compute_forward succeeded\n");
	if (dst->src[0] != nullptr && ggml_backend_buft_is_cuda_split(dst->src[0]->buffer->buft)) {
		ggml_cuda_set_peer_access(dst->src[1]->ne[1], ctx.device);
	}

	cudaEvent_t Event;
	cudaEventCreate(&Event);
	//FTaskDetector::GetInstance().SetIsInInferring(true);
	
	TArray<int> Input;
    FString Name;
	
	SCOPE_CYCLE_COUNTER(STAT_Inference);

	switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                GGML_LOG_ERROR("%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, dst->name, dst->src[0]->ne[3], dst->src[1]->ne[3]);
                return false;
            } else {
                ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV:
            ggml_cuda_op_rwkv_wkv(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        default:
            return false;
    }

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
		CUDA_CHECK(err);
	}

	cudaEventRecord(Event);
	(new FAutoDeleteAsyncTask<FMonitorGGMLOperatorTask>(Event))->StartBackgroundTask();

	//FTaskDetector::GetInstance().SetIsInInferring(false);
	
    return true;
}

#define LLAMA_LAYERS           32
#define LLAMA_NODES_LAYER_x    32
#define LLAMA_NODES_LM_HEAD     3
#define LLAMA_NODES_EXTRA_END   2
#define LLAMA_NODES            1029

uint32_t get_first_layer_num(ggml_cgraph * graph) {
    char * num = graph->nodes[0]->name + 5;
    uint32_t i = 0, sum = 0;
    while (num[i] != '\0') {
        sum *= 10;
        sum += num[i++] - '0';
    }
    return sum;
}

uint32_t get_layer_cnt(uint32_t n_nodes)
{
    return n_nodes / LLAMA_NODES_LAYER_x;
}

uint32_t get_layers_num(uint32_t idx_node, uint32_t n_nodes, uint32_t first_layer_num) {
    if ((n_nodes / LLAMA_NODES_LAYER_x) * LLAMA_NODES_LAYER_x == n_nodes)
    {
        return idx_node / LLAMA_NODES_LAYER_x + first_layer_num;
    }
    else
    {
        n_nodes -= LLAMA_NODES_LM_HEAD;
        int layer_num = idx_node / LLAMA_NODES_LAYER_x;
        if (n_nodes - layer_num * LLAMA_NODES_LAYER_x <= 2) return first_layer_num + layer_num - 1;
        else return first_layer_num + layer_num;
    }
}

bool get_cgraph_type(uint32_t layer_cnt)
{
    return layer_cnt != LLAMA_LAYERS;
}

enum ggml_status ggml_backend_cuda_graph_compute_scheduled(ggml_backend_t backend, ggml_cgraph * cgraph) {
	//fprintf(stderr, "register_ggml_backend_cuda_graph_compute_scheduled\n");
	//std::ofstream("debug.txt", std::ios::app) << "register_ggml_backend_cuda_graph_compute_scheduled" <<  std::endl;
	FTaskScheduler::GetInstance().SetAllGraphNodes(cgraph->n_nodes);
    uint32_t first_layer_num = get_first_layer_num(cgraph);
    uint32_t layer_cnt = get_layer_cnt(cgraph->n_nodes);
    bool is_skip_computing = get_cgraph_type(layer_cnt);
	FTaskScheduler::GetInstance().SetFirstLayerNum(first_layer_num);
    std::string skip_begin_node_type = "l_out-";

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

#ifdef USE_CUDA_GRAPH
    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

    // Objects required for CUDA Graph
    if (cuda_ctx->cuda_graph == nullptr) {
        cuda_ctx->cuda_graph.reset(new ggml_cuda_graph());
    }

    bool use_cuda_graph = true;
    bool cuda_graph_update_required = false;
    // vector of pointers to CUDA cpy kernels, which are required to identify
    // kernel parameters which need updated in the graph for each token
    std::vector<void *> ggml_cuda_cpy_fn_ptrs;

    if (cuda_ctx->cuda_graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < CC_AMPERE) {
            cuda_ctx->cuda_graph->disable_due_to_gpu_arch = true;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
#endif
        }
    }

    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
    // or previous graph capture failure.
    // Also disable for multi-gpu for now. TO DO investigate
    if (disable_cuda_graphs_due_to_env
        || cuda_ctx->cuda_graph->disable_due_to_gpu_arch
        || cuda_ctx->cuda_graph->disable_due_to_too_many_updates
        || cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture) {
        use_cuda_graph = false;
    }

    if (use_cuda_graph) {
        if (cuda_ctx->cuda_graph->instance == nullptr) {
            cuda_graph_update_required = true;
        }

        // Check if the graph size has changed
        if (cuda_ctx->cuda_graph->ggml_graph_properties.size() != (size_t)cgraph->n_nodes) {
            cuda_graph_update_required = true;
            cuda_ctx->cuda_graph->ggml_graph_properties.resize(cgraph->n_nodes);
        }

        // Loop over nodes in GGML graph to determine if CUDA graph update is required
        // and store properties to allow this comparison for the next token
        for (int i = 0; i < cgraph->n_nodes; i++) {
            bool has_matching_properties = true;
            if (!cuda_graph_update_required) {
                has_matching_properties = ggml_graph_node_has_matching_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
            }
            if (!has_matching_properties) {
                cuda_graph_update_required = true;
            }
            set_ggml_graph_node_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
        }

        // Loop over nodes in GGML graph to obtain info needed for CUDA graph
        cuda_ctx->cuda_graph->updated_kernel_arg.clear();
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];

            if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                continue;
            }

            if (node->src[0] && node->src[0]->buffer && ggml_backend_buft_is_cuda_split(node->src[0]->buffer->buft)) {
                use_cuda_graph = false; // Split buffers are not supported by CUDA graph capture
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to split buffer\n", __func__);
#endif
            }

            if (node->op == GGML_OP_MUL_MAT_ID) {
                use_cuda_graph = false; // This node type is not supported by CUDA graph capture
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to mul_mat_id\n", __func__);
#endif
            }

            if (node->op == GGML_OP_ADD && node->src[1] && node->src[1]->ne[1] > 1) {
                // disable CUDA graphs for batch size > 1 for now.
                // Changes in batch size or context size can cause changes to the grid size of some kernels.
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
#endif
            }

            if (node->op == GGML_OP_CPY) {
                // store the copy op parameter which changes with each token.
                cuda_ctx->cuda_graph->updated_kernel_arg.push_back((char **) &(node->src[1]->data));
                // store a pointer to each copy op CUDA kernel to identify it later
                void * ptr = ggml_cuda_cpy_fn(node->src[0], node->src[1]);
                if (!ptr) {
                    use_cuda_graph = false;
#ifndef NDEBUG
                    GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported copy op\n", __func__);
#endif
                } else {
                    if (std::find(ggml_cuda_cpy_fn_ptrs.begin(), ggml_cuda_cpy_fn_ptrs.end(), ptr) == ggml_cuda_cpy_fn_ptrs.end()) {
                        ggml_cuda_cpy_fn_ptrs.push_back(ptr);
                    }
                }
            }

            if (!use_cuda_graph) {
                break;
            }
        }

        // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
        if (use_cuda_graph && cuda_graph_update_required) {
            cuda_ctx->cuda_graph->number_consecutive_updates++;
        } else {
            cuda_ctx->cuda_graph->number_consecutive_updates = 0;
        }

        if (cuda_ctx->cuda_graph->number_consecutive_updates >= 4) {
            cuda_ctx->cuda_graph->disable_due_to_too_many_updates = true;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) { // Start CUDA graph capture
        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

#else
    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
#endif // USE_CUDA_GRAPH

    bool graph_evaluated_or_captured = false;

    bool skip_state = false;

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            for (int i = 0; i < cgraph->n_nodes; i++) {
                if (skip_state) break;

                ggml_tensor * node = cgraph->nodes[i];
            	uint32_t now_layer_num = get_layers_num( i + 1, cgraph->n_nodes, first_layer_num);
            	//std::ofstream("layer.txt", std::ios::app) << "graph nodes" << cgraph->n_nodes << "first layer" << first_layer_num << "idx" << i << "now layer" << now_layer_num << std::endl;
            	FTaskScheduler::GetInstance().SetNowLayerNum(now_layer_num);
            	FTaskScheduler::GetInstance().SetExeGraphNodes(i);

                if (is_skip_computing && !skip_state)
                {
                    if (now_layer_num != LLAMA_LAYERS - 1 && now_layer_num != LLAMA_LAYERS - 2) {
                        uint32_t skip_layer_num = FTaskScheduler::GetInstance().GetExitLayerNum();
                        std::string skip_begin_node_name = skip_begin_node_type + std::to_string(now_layer_num);
                        if (skip_layer_num > 0)
                        {
                        	if (skip_layer_num < now_layer_num)
                        	{
                        		skip_state = true;
                        		cgraph->last_skip_layers_num = skip_layer_num;
                        	}
                        	else if (skip_layer_num == now_layer_num)
                        	{
                        		if (!strcmp(node->name, skip_begin_node_name.c_str()))
                        		{
                        			skip_state = true;
                        			cgraph->last_skip_layers_num = skip_layer_num;
                        		}
                        	}
                        }
                    }
                }

                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }

#ifndef NDEBUG
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               ggml_backend_buft_is_cuda_split(node->src[j]->buffer->buft));
                    }
                }
#endif
                bool ok = ggml_cuda_compute_forward_scheduled(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
            	
            }
            if (!skip_state) {
                cgraph->last_skip_layers_num = get_layers_num(cgraph->n_nodes - 1, cgraph->n_nodes, first_layer_num);
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (cuda_ctx->cuda_graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(cuda_ctx->cuda_graph->graph));
                cuda_ctx->cuda_graph->graph = nullptr;
            }
            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_ctx->cuda_graph->graph));

#if 0
            if (disable_cuda_graphs_due_to_failed_capture) {
                use_cuda_graph = false;
                cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture = true;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to failed graph capture\n", __func__);
#endif
            } else {
                graph_evaluated_or_captured = true; // CUDA graph has been captured
            }
#endif
            graph_evaluated_or_captured = true; // CUDA graph has been captured
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        if (cuda_ctx->cuda_graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
        }

        // Perform update to graph (if required for this token), and change copy parameter (required for every token)

        if (cuda_graph_update_required) {
            // Extract nodes from graph
            // First call with null argument gets number of nodes in graph
            CUDA_CHECK(cudaGraphGetNodes(cuda_ctx->cuda_graph->graph, nullptr, &cuda_ctx->cuda_graph->num_nodes));
            // Subsequent call with non-null argument gets nodes
            cuda_ctx->cuda_graph->nodes.clear();
            cuda_ctx->cuda_graph->nodes.resize(cuda_ctx->cuda_graph->num_nodes);
            cuda_ctx->cuda_graph->params.clear();
            cuda_ctx->cuda_graph->params.resize(cuda_ctx->cuda_graph->num_nodes);
            if (cuda_ctx->cuda_graph->num_nodes > 0) {
                CUDA_CHECK(cudaGraphGetNodes(cuda_ctx->cuda_graph->graph, cuda_ctx->cuda_graph->nodes.data(), &cuda_ctx->cuda_graph->num_nodes));

                // Loop over nodes, and extract kernel parameters from each node
                for (size_t i = 0; i < cuda_ctx->cuda_graph->num_nodes; i++) {
                    cudaGraphNodeType node_type;
                    CUDA_CHECK(cudaGraphNodeGetType(cuda_ctx->cuda_graph->nodes[i], &node_type));
                    if (node_type == cudaGraphNodeTypeKernel) {
                        cudaError_t stat = cudaGraphKernelNodeGetParams(cuda_ctx->cuda_graph->nodes[i], &cuda_ctx->cuda_graph->params[i]); // Get params using runtime
                        if (stat == cudaErrorInvalidDeviceFunction) {
                            // Fails due to incorrect handling by CUDA runtime of CUDA BLAS node.
                            // We don't need to update blas nodes, so clear error and move on.
                            cudaGetLastError();
                        } else {
                            GGML_ASSERT(stat == cudaSuccess);
                        }
                    }
                }
            }
        }

        // One of the arguments to the copy kernel is updated for each token, hence we need to
        // replace that argument with the updated value in the CUDA graph
        if (!cuda_graph_update_required) { // on update steps, the live parameters will already be captured
            int k = 0;
            for (size_t i = 0; i < cuda_ctx->cuda_graph->num_nodes; i++) {
                if(count(ggml_cuda_cpy_fn_ptrs.begin(), ggml_cuda_cpy_fn_ptrs.end(), cuda_ctx->cuda_graph->params[i].func) > 0) {
                    char ** updated_kernel_arg_ptr = cuda_ctx->cuda_graph->updated_kernel_arg.at(k++);
                    cuda_ctx->cuda_graph->params[i].kernelParams[1] = updated_kernel_arg_ptr;
                    CUDA_CHECK(cudaGraphKernelNodeSetParams(cuda_ctx->cuda_graph->nodes[i], &cuda_ctx->cuda_graph->params[i]));
                }
            }
        }

        // Update graph executable
        cudaGraphExecUpdateResultInfo result_info;
        cudaError_t stat = cudaGraphExecUpdate(cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, &result_info);
        if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif
            // The pre-existing graph exec cannot be updated due to violated constraints
            // so instead clear error and re-instantiate
            cudaGetLastError();
            CUDA_CHECK(cudaGraphExecDestroy(cuda_ctx->cuda_graph->instance));
            cuda_ctx->cuda_graph->instance = nullptr;
            CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
        } else {
            GGML_ASSERT(stat == cudaSuccess);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(cuda_ctx->cuda_graph->instance, cuda_ctx->stream()));
#else
        graph_evaluated_or_captured = true;
#endif // USE_CUDA_GRAPH
    }

    return GGML_STATUS_SUCCESS;
}

int layer_count = 0;
int extra_count = 0;
int frame_count = -1;

enum ggml_status ggml_backend_cuda_graph_compute_default(ggml_backend_t backend, ggml_cgraph * cgraph) {

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

#ifdef USE_CUDA_GRAPH
    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

    // Objects required for CUDA Graph
    if (cuda_ctx->cuda_graph == nullptr) {
        cuda_ctx->cuda_graph.reset(new ggml_cuda_graph());
    }

    bool use_cuda_graph = true;
    bool cuda_graph_update_required = false;
    // vector of pointers to CUDA cpy kernels, which are required to identify
    // kernel parameters which need updated in the graph for each token
    std::vector<void *> ggml_cuda_cpy_fn_ptrs;

    if (cuda_ctx->cuda_graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < CC_AMPERE) {
            cuda_ctx->cuda_graph->disable_due_to_gpu_arch = true;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
#endif
        }
    }

    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
    // or previous graph capture failure.
    // Also disable for multi-gpu for now. TO DO investigate
    if (disable_cuda_graphs_due_to_env
        || cuda_ctx->cuda_graph->disable_due_to_gpu_arch
        || cuda_ctx->cuda_graph->disable_due_to_too_many_updates
        || cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture) {
        use_cuda_graph = false;
    }

    if (use_cuda_graph) {
        if (cuda_ctx->cuda_graph->instance == nullptr) {
            cuda_graph_update_required = true;
        }

        // Check if the graph size has changed
        if (cuda_ctx->cuda_graph->ggml_graph_properties.size() != (size_t)cgraph->n_nodes) {
            cuda_graph_update_required = true;
            cuda_ctx->cuda_graph->ggml_graph_properties.resize(cgraph->n_nodes);
        }

        // Loop over nodes in GGML graph to determine if CUDA graph update is required
        // and store properties to allow this comparison for the next token
        for (int i = 0; i < cgraph->n_nodes; i++) {
            bool has_matching_properties = true;
            if (!cuda_graph_update_required) {
                has_matching_properties = ggml_graph_node_has_matching_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
            }
            if (!has_matching_properties) {
                cuda_graph_update_required = true;
            }
            set_ggml_graph_node_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
        }

        // Loop over nodes in GGML graph to obtain info needed for CUDA graph
        cuda_ctx->cuda_graph->updated_kernel_arg.clear();
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];

            if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                continue;
            }

            if (node->src[0] && node->src[0]->buffer && ggml_backend_buft_is_cuda_split(node->src[0]->buffer->buft)) {
                use_cuda_graph = false; // Split buffers are not supported by CUDA graph capture
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to split buffer\n", __func__);
#endif
            }

            if (node->op == GGML_OP_MUL_MAT_ID) {
                use_cuda_graph = false; // This node type is not supported by CUDA graph capture
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to mul_mat_id\n", __func__);
#endif
            }

            if (node->op == GGML_OP_ADD && node->src[1] && node->src[1]->ne[1] > 1) {
                // disable CUDA graphs for batch size > 1 for now.
                // Changes in batch size or context size can cause changes to the grid size of some kernels.
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
#endif
            }

            if (node->op == GGML_OP_CPY) {
                // store the copy op parameter which changes with each token.
                cuda_ctx->cuda_graph->updated_kernel_arg.push_back((char **) &(node->src[1]->data));
                // store a pointer to each copy op CUDA kernel to identify it later
                void * ptr = ggml_cuda_cpy_fn(node->src[0], node->src[1]);
                if (!ptr) {
                    use_cuda_graph = false;
#ifndef NDEBUG
                    GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported copy op\n", __func__);
#endif
                } else {
                    if (std::find(ggml_cuda_cpy_fn_ptrs.begin(), ggml_cuda_cpy_fn_ptrs.end(), ptr) == ggml_cuda_cpy_fn_ptrs.end()) {
                        ggml_cuda_cpy_fn_ptrs.push_back(ptr);
                    }
                }
            }

            if (!use_cuda_graph) {
                break;
            }
        }

        // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
        if (use_cuda_graph && cuda_graph_update_required) {
            cuda_ctx->cuda_graph->number_consecutive_updates++;
        } else {
            cuda_ctx->cuda_graph->number_consecutive_updates = 0;
        }

        if (cuda_ctx->cuda_graph->number_consecutive_updates >= 4) {
            cuda_ctx->cuda_graph->disable_due_to_too_many_updates = true;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) { // Start CUDA graph capture
        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

#else
    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
#endif // USE_CUDA_GRAPH

    bool graph_evaluated_or_captured = false;

	bool scheduled_flag = false;
	bool exit_flag = false;

	int window_length = FPerformanceDetector::GetInstance().GetWindowsLength();
	double inference_time_per_action_3b = 185;
	double inference_time_per_layer = 0.3;
	double inference_time_per_window = inference_time_per_action_3b / window_length;
	int scheduled_count = (inference_time_per_window / inference_time_per_layer) * 1;
	FTaskDetector::GetInstance().SetScheduledCount(scheduled_count);
	FTaskDetector::GetInstance().SetIsInGraphComputing(true);
	
    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            for (int i = 0; i < cgraph->n_nodes; i++)
            {
	            ggml_tensor * node = cgraph->nodes[i];

	            if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
	            	continue;
	            }

#ifndef NDEBUG
	            assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device));
	            for (int j = 0; j < GGML_MAX_SRC; j++) {
	            	if (node->src[j] != nullptr) {
	            		assert(node->src[j]->buffer);
	            		assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
							   ggml_backend_buft_is_cuda_split(node->src[j]->buffer->buft));
	            	}
	            }
            	
#endif
	            /*int OpPolicy = FTaskScheduler::GetInstance().GetOpPolicy();
            	if (OpPolicy == 1 && FPerformanceDetector::GetInstance().GetTargetAPM() != 300)
            	{
            		FTaskScheduler::GetInstance().WaitForRenderingTaskCompletion();
            	}
	            if (OpPolicy == 1/* && FPerformanceDetector::GetInstance().GetTargetAPM() == 300#1#)
	            {
	            	if (layer_count != 0)
	            		FTaskScheduler::GetInstance().WaitForRenderingTaskCompletion();
	            	if (!scheduled_flag)
	            	{
	            		FTaskScheduler::GetInstance().WaitForRenderingTaskCompletion();
	            		scheduled_flag = true;
	            	}
	            }*/

            	FTaskScheduler::GetInstance().WaitForRenderingTaskCompletion();

	            //FTaskDetector::GetInstance().SetIsInInferring(true);
	            bool ok = ggml_cuda_compute_forward_scheduled(*cuda_ctx, node);
	            if (!ok) {
	            	GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
	            }
	            GGML_ASSERT(ok);
	            std::string node_name(node->name);
            	
	            /*if (OpPolicy == 1/* && FPerformanceDetector::GetInstance().GetTargetAPM() == 300#1#)
	            {
	            	if (node_name.find("l_out") != std::string::npos)
	            	{
	            		layer_count++;
	            		if (layer_count == scheduled_count) layer_count = 0;
	            		FTaskDetector::GetInstance().SetNodeCount(layer_count);
	            		
	            	}
	            	/*if (FTaskDetector::GetInstance().GetIsPrefilling())
					{#1#
	            	/*cudaEvent_t Event;
					cudaEventCreate(&Event);
					cudaEventRecord(Event);
					(new FAutoDeleteAsyncTask<FMonitorGGMLOperatorTask>(Event))->StartBackgroundTask();#1#
	            	//cudaDeviceSynchronize();
	            	//}
	            }*/
	            //FTaskDetector::GetInstance().SetIsInInferring(false);
	            
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (cuda_ctx->cuda_graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(cuda_ctx->cuda_graph->graph));
                cuda_ctx->cuda_graph->graph = nullptr;
            }
            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_ctx->cuda_graph->graph));

#if 0
            if (disable_cuda_graphs_due_to_failed_capture) {
                use_cuda_graph = false;
                cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture = true;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to failed graph capture\n", __func__);
#endif
            } else {
                graph_evaluated_or_captured = true; // CUDA graph has been captured
            }
#endif
            graph_evaluated_or_captured = true; // CUDA graph has been captured
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        if (cuda_ctx->cuda_graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
        }

        // Perform update to graph (if required for this token), and change copy parameter (required for every token)

        if (cuda_graph_update_required) {
            // Extract nodes from graph
            // First call with null argument gets number of nodes in graph
            CUDA_CHECK(cudaGraphGetNodes(cuda_ctx->cuda_graph->graph, nullptr, &cuda_ctx->cuda_graph->num_nodes));
            // Subsequent call with non-null argument gets nodes
            cuda_ctx->cuda_graph->nodes.clear();
            cuda_ctx->cuda_graph->nodes.resize(cuda_ctx->cuda_graph->num_nodes);
            cuda_ctx->cuda_graph->params.clear();
            cuda_ctx->cuda_graph->params.resize(cuda_ctx->cuda_graph->num_nodes);
            if (cuda_ctx->cuda_graph->num_nodes > 0) {
                CUDA_CHECK(cudaGraphGetNodes(cuda_ctx->cuda_graph->graph, cuda_ctx->cuda_graph->nodes.data(), &cuda_ctx->cuda_graph->num_nodes));

                // Loop over nodes, and extract kernel parameters from each node
                for (size_t i = 0; i < cuda_ctx->cuda_graph->num_nodes; i++) {
                    cudaGraphNodeType node_type;
                    CUDA_CHECK(cudaGraphNodeGetType(cuda_ctx->cuda_graph->nodes[i], &node_type));
                    if (node_type == cudaGraphNodeTypeKernel) {
                        cudaError_t stat = cudaGraphKernelNodeGetParams(cuda_ctx->cuda_graph->nodes[i], &cuda_ctx->cuda_graph->params[i]); // Get params using runtime
                        if (stat == cudaErrorInvalidDeviceFunction) {
                            // Fails due to incorrect handling by CUDA runtime of CUDA BLAS node.
                            // We don't need to update blas nodes, so clear error and move on.
                            cudaGetLastError();
                        } else {
                            GGML_ASSERT(stat == cudaSuccess);
                        }
                    }
                }
            }
        }

        // One of the arguments to the copy kernel is updated for each token, hence we need to
        // replace that argument with the updated value in the CUDA graph
        if (!cuda_graph_update_required) { // on update steps, the live parameters will already be captured
            int k = 0;
            for (size_t i = 0; i < cuda_ctx->cuda_graph->num_nodes; i++) {
                if(count(ggml_cuda_cpy_fn_ptrs.begin(), ggml_cuda_cpy_fn_ptrs.end(), cuda_ctx->cuda_graph->params[i].func) > 0) {
                    char ** updated_kernel_arg_ptr = cuda_ctx->cuda_graph->updated_kernel_arg.at(k++);
                    cuda_ctx->cuda_graph->params[i].kernelParams[1] = updated_kernel_arg_ptr;
                    CUDA_CHECK(cudaGraphKernelNodeSetParams(cuda_ctx->cuda_graph->nodes[i], &cuda_ctx->cuda_graph->params[i]));
                }
            }
        }

        // Update graph executable
        cudaGraphExecUpdateResultInfo result_info;
        cudaError_t stat = cudaGraphExecUpdate(cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, &result_info);
        if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif
            // The pre-existing graph exec cannot be updated due to violated constraints
            // so instead clear error and re-instantiate
            cudaGetLastError();
            CUDA_CHECK(cudaGraphExecDestroy(cuda_ctx->cuda_graph->instance));
            cuda_ctx->cuda_graph->instance = nullptr;
            CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
        } else {
            GGML_ASSERT(stat == cudaSuccess);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(cuda_ctx->cuda_graph->instance, cuda_ctx->stream()));
#else
        graph_evaluated_or_captured = true;
#endif // USE_CUDA_GRAPH
    }

	FTaskDetector::GetInstance().SetIsInGraphComputing(false);

    return GGML_STATUS_SUCCESS;
}


enum ggml_status ggml_backend_cuda_graph_compute_scheduled_v2(ggml_backend_t backend, ggml_cgraph * cgraph) {

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

#ifdef USE_CUDA_GRAPH
    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

    // Objects required for CUDA Graph
    if (cuda_ctx->cuda_graph == nullptr) {
        cuda_ctx->cuda_graph.reset(new ggml_cuda_graph());
    }

    bool use_cuda_graph = true;
    bool cuda_graph_update_required = false;
    // vector of pointers to CUDA cpy kernels, which are required to identify
    // kernel parameters which need updated in the graph for each token
    std::vector<void *> ggml_cuda_cpy_fn_ptrs;

    if (cuda_ctx->cuda_graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < CC_AMPERE) {
            cuda_ctx->cuda_graph->disable_due_to_gpu_arch = true;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
#endif
        }
    }

    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
    // or previous graph capture failure.
    // Also disable for multi-gpu for now. TO DO investigate
    if (disable_cuda_graphs_due_to_env
        || cuda_ctx->cuda_graph->disable_due_to_gpu_arch
        || cuda_ctx->cuda_graph->disable_due_to_too_many_updates
        || cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture) {
        use_cuda_graph = false;
    }

    if (use_cuda_graph) {
        if (cuda_ctx->cuda_graph->instance == nullptr) {
            cuda_graph_update_required = true;
        }

        // Check if the graph size has changed
        if (cuda_ctx->cuda_graph->ggml_graph_properties.size() != (size_t)cgraph->n_nodes) {
            cuda_graph_update_required = true;
            cuda_ctx->cuda_graph->ggml_graph_properties.resize(cgraph->n_nodes);
        }

        // Loop over nodes in GGML graph to determine if CUDA graph update is required
        // and store properties to allow this comparison for the next token
        for (int i = 0; i < cgraph->n_nodes; i++) {
            bool has_matching_properties = true;
            if (!cuda_graph_update_required) {
                has_matching_properties = ggml_graph_node_has_matching_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
            }
            if (!has_matching_properties) {
                cuda_graph_update_required = true;
            }
            set_ggml_graph_node_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
        }

        // Loop over nodes in GGML graph to obtain info needed for CUDA graph
        cuda_ctx->cuda_graph->updated_kernel_arg.clear();
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];

            if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                continue;
            }

            if (node->src[0] && node->src[0]->buffer && ggml_backend_buft_is_cuda_split(node->src[0]->buffer->buft)) {
                use_cuda_graph = false; // Split buffers are not supported by CUDA graph capture
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to split buffer\n", __func__);
#endif
            }

            if (node->op == GGML_OP_MUL_MAT_ID) {
                use_cuda_graph = false; // This node type is not supported by CUDA graph capture
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to mul_mat_id\n", __func__);
#endif
            }

            if (node->op == GGML_OP_ADD && node->src[1] && node->src[1]->ne[1] > 1) {
                // disable CUDA graphs for batch size > 1 for now.
                // Changes in batch size or context size can cause changes to the grid size of some kernels.
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
#endif
            }

            if (node->op == GGML_OP_CPY) {
                // store the copy op parameter which changes with each token.
                cuda_ctx->cuda_graph->updated_kernel_arg.push_back((char **) &(node->src[1]->data));
                // store a pointer to each copy op CUDA kernel to identify it later
                void * ptr = ggml_cuda_cpy_fn(node->src[0], node->src[1]);
                if (!ptr) {
                    use_cuda_graph = false;
#ifndef NDEBUG
                    GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported copy op\n", __func__);
#endif
                } else {
                    if (std::find(ggml_cuda_cpy_fn_ptrs.begin(), ggml_cuda_cpy_fn_ptrs.end(), ptr) == ggml_cuda_cpy_fn_ptrs.end()) {
                        ggml_cuda_cpy_fn_ptrs.push_back(ptr);
                    }
                }
            }

            if (!use_cuda_graph) {
                break;
            }
        }

        // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
        if (use_cuda_graph && cuda_graph_update_required) {
            cuda_ctx->cuda_graph->number_consecutive_updates++;
        } else {
            cuda_ctx->cuda_graph->number_consecutive_updates = 0;
        }

        if (cuda_ctx->cuda_graph->number_consecutive_updates >= 4) {
            cuda_ctx->cuda_graph->disable_due_to_too_many_updates = true;
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) { // Start CUDA graph capture
        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

#else
    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
#endif // USE_CUDA_GRAPH

    bool graph_evaluated_or_captured = false;

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            for (int i = 0; i < cgraph->n_nodes; i++) {

                ggml_tensor * node = cgraph->nodes[i];

                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }

#ifndef NDEBUG
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               ggml_backend_buft_is_cuda_split(node->src[j]->buffer->buft));
                    }
                }
#endif
                bool ok = ggml_cuda_compute_forward_scheduled(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
            	
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (cuda_ctx->cuda_graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(cuda_ctx->cuda_graph->graph));
                cuda_ctx->cuda_graph->graph = nullptr;
            }
            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_ctx->cuda_graph->graph));

#if 0
            if (disable_cuda_graphs_due_to_failed_capture) {
                use_cuda_graph = false;
                cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture = true;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to failed graph capture\n", __func__);
#endif
            } else {
                graph_evaluated_or_captured = true; // CUDA graph has been captured
            }
#endif
            graph_evaluated_or_captured = true; // CUDA graph has been captured
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        if (cuda_ctx->cuda_graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
        }

        // Perform update to graph (if required for this token), and change copy parameter (required for every token)

        if (cuda_graph_update_required) {
            // Extract nodes from graph
            // First call with null argument gets number of nodes in graph
            CUDA_CHECK(cudaGraphGetNodes(cuda_ctx->cuda_graph->graph, nullptr, &cuda_ctx->cuda_graph->num_nodes));
            // Subsequent call with non-null argument gets nodes
            cuda_ctx->cuda_graph->nodes.clear();
            cuda_ctx->cuda_graph->nodes.resize(cuda_ctx->cuda_graph->num_nodes);
            cuda_ctx->cuda_graph->params.clear();
            cuda_ctx->cuda_graph->params.resize(cuda_ctx->cuda_graph->num_nodes);
            if (cuda_ctx->cuda_graph->num_nodes > 0) {
                CUDA_CHECK(cudaGraphGetNodes(cuda_ctx->cuda_graph->graph, cuda_ctx->cuda_graph->nodes.data(), &cuda_ctx->cuda_graph->num_nodes));

                // Loop over nodes, and extract kernel parameters from each node
                for (size_t i = 0; i < cuda_ctx->cuda_graph->num_nodes; i++) {
                    cudaGraphNodeType node_type;
                    CUDA_CHECK(cudaGraphNodeGetType(cuda_ctx->cuda_graph->nodes[i], &node_type));
                    if (node_type == cudaGraphNodeTypeKernel) {
                        cudaError_t stat = cudaGraphKernelNodeGetParams(cuda_ctx->cuda_graph->nodes[i], &cuda_ctx->cuda_graph->params[i]); // Get params using runtime
                        if (stat == cudaErrorInvalidDeviceFunction) {
                            // Fails due to incorrect handling by CUDA runtime of CUDA BLAS node.
                            // We don't need to update blas nodes, so clear error and move on.
                            cudaGetLastError();
                        } else {
                            GGML_ASSERT(stat == cudaSuccess);
                        }
                    }
                }
            }
        }

        // One of the arguments to the copy kernel is updated for each token, hence we need to
        // replace that argument with the updated value in the CUDA graph
        if (!cuda_graph_update_required) { // on update steps, the live parameters will already be captured
            int k = 0;
            for (size_t i = 0; i < cuda_ctx->cuda_graph->num_nodes; i++) {
                if(count(ggml_cuda_cpy_fn_ptrs.begin(), ggml_cuda_cpy_fn_ptrs.end(), cuda_ctx->cuda_graph->params[i].func) > 0) {
                    char ** updated_kernel_arg_ptr = cuda_ctx->cuda_graph->updated_kernel_arg.at(k++);
                    cuda_ctx->cuda_graph->params[i].kernelParams[1] = updated_kernel_arg_ptr;
                    CUDA_CHECK(cudaGraphKernelNodeSetParams(cuda_ctx->cuda_graph->nodes[i], &cuda_ctx->cuda_graph->params[i]));
                }
            }
        }

        // Update graph executable
        cudaGraphExecUpdateResultInfo result_info;
        cudaError_t stat = cudaGraphExecUpdate(cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, &result_info);
        if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif
            // The pre-existing graph exec cannot be updated due to violated constraints
            // so instead clear error and re-instantiate
            cudaGetLastError();
            CUDA_CHECK(cudaGraphExecDestroy(cuda_ctx->cuda_graph->instance));
            cuda_ctx->cuda_graph->instance = nullptr;
            CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
        } else {
            GGML_ASSERT(stat == cudaSuccess);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(cuda_ctx->cuda_graph->instance, cuda_ctx->stream()));
#else
        graph_evaluated_or_captured = true;
#endif // USE_CUDA_GRAPH
    }

    return GGML_STATUS_SUCCESS;
}