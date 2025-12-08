#pragma once

#include "CoreMinimal.h"
#include <string>
#include "../Private/llamacpp/llama.h"
#include "../../SchedulingSystem/PerformanceDetector/Public/PerformanceDetector.h"
#include "../../SchedulingSystem/TaskDetector/Public/TaskDetector.h"

struct llama_sampler;

class DLLEXPORT FNewLlamaModel
{
public:
	static FNewLlamaModel& GetInstance();
	void Init();
	void Exit();
	void Inference(std::string prompt, FString* content, int time_per_action);
	float GetInferenceTime();
	float GetTotalInferenceTime();
	bool CheckAutoUpdateSkipLayers();
	std::string prompt;

private:
	FNewLlamaModel() {};
	FNewLlamaModel(const FNewLlamaModel&) = delete;
	FNewLlamaModel& operator=(const FNewLlamaModel&) = delete;
	
	llama_model_params model_params;
	llama_model* model;
	llama_context_params ctx_params;
	llama_context* ctx;
	llama_sampler* smpl;
	int n_len;
	float AvgInferenceTime;
	float TotalInferenceTime;
	int model_type;
	double decode_time_per_token;
	double decode_time_per_layer;
	double prefill_time_per_token;
	double prefill_time_per_layer;
};










