#pragma once

#include "CoreMinimal.h"
#include "../Private/llamacpp/common/common.h"
#include "../Private/llamacpp/llama.h"
#include "../../SchedulingSystem/PerformanceDetector/Public/PerformanceDetector.h"
#include "../../SchedulingSystem/TaskDetector/Public/TaskDetector.h"

class DLLEXPORT FLlamaModel
{
public:
	static FLlamaModel& GetInstance();
	void Init();
	void Exit();
	void Inference(std::string prompt, FString* content, bool AutoSchedule);
	float GetInferenceSpeed();
	std::string prompt;

private:
	FLlamaModel() {};
	FLlamaModel(const FLlamaModel&) = delete;
	FLlamaModel& operator=(const FLlamaModel&) = delete;
	
	gpt_params params;
	llama_model_params model_params;
	llama_model* model;
	llama_context_params ctx_params;
	llama_context* ctx;
	int n_len;
	float InferenceSpeed;
};










