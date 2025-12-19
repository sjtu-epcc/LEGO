#pragma once
#include "CoreMinimal.h"
#include "ggml-cuda.h"
#include "../../../SchedulingSystem/TaskScheduler/Public/TaskScheduler.h"

GGML_CALL bool ggml_cuda_compute_forward_scheduled(ggml_compute_params * params, ggml_tensor * tensor) {
	
	//printf("register succeeded\n");
    if (!ggml_cublas_loaded()) return false;

	
    ggml_cuda_func_t func;
	TArray<int> Input;
    FString Name;


    const bool any_on_device = tensor->backend == GGML_BACKEND_GPU
        || (tensor->src[0] != nullptr && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT))
        || (tensor->src[1] != nullptr && tensor->src[1]->backend == GGML_BACKEND_GPU);

    if (!any_on_device && tensor->op != GGML_OP_MUL_MAT && tensor->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT) {
        if (tensor->src[0]->ne[3] != tensor->src[1]->ne[3]) {
#ifndef NDEBUG
            fprintf(stderr, "%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, tensor->name, tensor->src[0]->ne[3], tensor->src[1]->ne[3]);
#endif
            return false;
        }
    }

    switch (tensor->op) {
        case GGML_OP_REPEAT:
            func = ggml_cuda_repeat;
            Name = TEXT("repeat");
            break;
        case GGML_OP_GET_ROWS:
            func = ggml_cuda_get_rows;
    		Name = TEXT("get_rows");
            break;
        case GGML_OP_DUP:
            func = ggml_cuda_dup;
    		Name = TEXT("dup");
            break;
        case GGML_OP_ADD:
            func = ggml_cuda_add;
    		Name = TEXT("add");
            break;
        case GGML_OP_ACC:
            func = ggml_cuda_acc;
    		Name = TEXT("acc");
            break;
        case GGML_OP_MUL:
            func = ggml_cuda_mul;
    		Name = TEXT("mul");
            break;
        case GGML_OP_DIV:
            func = ggml_cuda_div;
    		Name = TEXT("div");
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(tensor)) {
                case GGML_UNARY_OP_GELU:
                    func = ggml_cuda_gelu;
            		Name = TEXT("gelu");
                    break;
                case GGML_UNARY_OP_SILU:
                    func = ggml_cuda_silu;
            		Name = TEXT("silu");
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    func = ggml_cuda_gelu_quick;
            		Name = TEXT("gelu_quick");
                    break;
                case GGML_UNARY_OP_TANH:
                    func = ggml_cuda_tanh;
            		Name = TEXT("tanh");
                    break;
                case GGML_UNARY_OP_RELU:
                    func = ggml_cuda_relu;
            		Name = TEXT("relu");
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    func = ggml_cuda_hardsigmoid;
            		Name = TEXT("hardsigmoid");
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    func = ggml_cuda_hardswish;
            		Name = TEXT("hardswish");
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            func = ggml_cuda_norm;
    		Name = TEXT("norm");
            break;
        case GGML_OP_GROUP_NORM:
            func = ggml_cuda_group_norm;
    		Name = TEXT("group_norm");
            break;
        case GGML_OP_CONCAT:
            func = ggml_cuda_concat;
    		Name = TEXT("concat");
            break;
        case GGML_OP_UPSCALE:
            func = ggml_cuda_upscale;
    		Name = TEXT("upscale");
            break;
        case GGML_OP_PAD:
            func = ggml_cuda_pad;
    		Name = TEXT("pad");
            break;
        case GGML_OP_LEAKY_RELU:
            func = ggml_cuda_leaky_relu;
    		Name = TEXT("leaky_relu");
            break;
        case GGML_OP_RMS_NORM:
            func = ggml_cuda_rms_norm;
    		Name = TEXT("rms_norm");
            break;
        case GGML_OP_MUL_MAT:
            if (!any_on_device && !ggml_cuda_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_cuda_mul_mat;
    		Name = TEXT("mul_mat");
            break;
        case GGML_OP_MUL_MAT_ID:
            if (!any_on_device && !ggml_cuda_can_mul_mat(tensor->src[2], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_cuda_mul_mat_id;
    		Name = TEXT("mul_mat_id");
            break;
        case GGML_OP_SCALE:
            func = ggml_cuda_scale;
    		Name = TEXT("scale");
            break;
        case GGML_OP_SQR:
            func = ggml_cuda_sqr;
    		Name = TEXT("sqr");
            break;
        case GGML_OP_CLAMP:
            func = ggml_cuda_clamp;
    		Name = TEXT("clamp");
            break;
        case GGML_OP_CPY:
            func = ggml_cuda_cpy;
    		Name = TEXT("cpy");
            break;
        case GGML_OP_CONT:
            func = ggml_cuda_dup;
    		Name = TEXT("dup");
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            func = ggml_cuda_nop;
    		Name = TEXT("nop");
            break;
        case GGML_OP_DIAG_MASK_INF:
            func = ggml_cuda_diag_mask_inf;
    		Name = TEXT("diag_mask_inf");
            break;
        case GGML_OP_SOFT_MAX:
            func = ggml_cuda_soft_max;
    		Name = TEXT("soft_max");
            break;
        case GGML_OP_ROPE:
            func = ggml_cuda_rope;
    		Name = TEXT("rope");
            break;
        case GGML_OP_ALIBI:
            func = ggml_cuda_alibi;
    		Name = TEXT("alibi");
            break;
        case GGML_OP_IM2COL:
            func = ggml_cuda_im2col;
    		Name = TEXT("im2col");
            break;
        case GGML_OP_POOL_2D:
            func = ggml_cuda_pool2d;
    		Name = TEXT("pool2d");
            break;
        case GGML_OP_SUM_ROWS:
            func = ggml_cuda_sum_rows;
    		Name = TEXT("sum_rows");
            break;
        case GGML_OP_ARGSORT:
            func = ggml_cuda_argsort;
    		Name = TEXT("argsort");
            break;
        default:
            return false;
    }

    if (tensor->src[0] != nullptr && tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT) {
        ggml_cuda_set_peer_access(tensor->src[1]->ne[1]);
    }

    if (params->ith != 0) {
        return true;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return true;
    }

	for (int i = 0; i < 4; ++i) Input.Push(tensor->src[0]->ne[i]);
	if (tensor->src[1] != nullptr) for (int i = 0; i < 4; ++i) Input.Push(tensor->src[1]->ne[i]);
	for (int i = 0; i < 4; ++i) Input.Push(tensor->ne[i]);
	
	
	FTaskScheduler::GetInstance().WaitForRenderingTaskCompletion();
	
	func(tensor->src[0], tensor->src[1], tensor);
	
	//FTaskDetector::GetInstance().UpdateGgmlCount();
	
    return true;
}