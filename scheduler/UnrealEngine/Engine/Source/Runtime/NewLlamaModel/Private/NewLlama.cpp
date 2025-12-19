#include "..\Public\NewLlama.h"
#include <chrono>
#include <thread>
#include "Editor/UnrealEdEngine.h"
#include "llamacpp/ggml-cuda.h"

DECLARE_STATS_GROUP(TEXT("Inference"), STATGROUP_LLAMA, STATCAT_Advanced);
DECLARE_CYCLE_STAT_EXTERN(TEXT("Simple time"), STAT_Simple, STATGROUP_LLAMA, );
DECLARE_CYCLE_STAT_EXTERN(TEXT("One Token time"), STAT_OneToken, STATGROUP_LLAMA, );
DEFINE_STAT(STAT_OneToken);
DEFINE_STAT(STAT_Simple);

FNewLlamaModel& FNewLlamaModel::GetInstance()
{
	static FNewLlamaModel llama;
	return llama;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 定义生成序列的函数
int* generate_sequence(int layers[], double probabilities[], int num_layers, int sequence_length) {
	// 计算累积概率
	double cumulative_probabilities[8];
	cumulative_probabilities[0] = probabilities[0];
	for (int i = 1; i < num_layers; i++) {
		cumulative_probabilities[i] = cumulative_probabilities[i - 1] + probabilities[i];
	}

	// 初始化随机数种子
	srand(time(NULL));

	// 动态分配内存存储序列
	int* sequence = (int*)malloc(sequence_length * sizeof(int));
	if (sequence == NULL) {
		fprintf(stderr, "内存分配失败\n");
		exit(1);
	}

	// 生成序列
	for (int i = 0; i < sequence_length; i++) {
		double random_value = (double)rand() / RAND_MAX; // 生成 [0, 1) 之间的随机数

		// 根据随机数选择层数
		for (int j = 0; j < num_layers; j++) {
			if (random_value < cumulative_probabilities[j]) {
				sequence[i] = layers[j];
				break;
			}
		}
	}

	return sequence;
}

void FNewLlamaModel::Init()
{
	fprintf(stderr, "FNewLlamaModel::Init()\n");
	std::string model_path;
	
	model_path = "E:\\Program Files\\Llama-3.2-8B-Instruct.Q8_0.gguf";
	n_len = 16;
	
	llama_backend_init();

	std::string type = "8B";
	if (model_path.find(type) == std::string::npos) model_type = 1;
	else model_type = 0;
	
	model_params = llama_model_default_params();
	model_params.n_gpu_layers = 10000;
	model = llama_load_model_from_file(model_path.c_str(), model_params);
	
	ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 2048;
	ctx_params.no_perf = false;
	
	decode_time_per_layer = 0.4f;
	decode_time_per_token = 13.5f;

	prefill_time_per_layer = 1.5f;
	prefill_time_per_token = 48.0f;

	register_ggml_backend_cuda_graph_compute(ggml_backend_cuda_graph_compute_default);
}

void FNewLlamaModel::Exit()
{
	llama_free(ctx);
	llama_free_model(model);

	fprintf(stderr, "Llama has exited.\n");
}

bool FNewLlamaModel::CheckAutoUpdateSkipLayers()
{
	int SkipPolicy = FTaskScheduler::GetInstance().GetSkipPolicy();
	int OpPolicy = FTaskScheduler::GetInstance().GetOpPolicy();
	if (!model_type && !SkipPolicy && (OpPolicy == 0 || OpPolicy == 1)) return true;
	return false;
}

void FNewLlamaModel::Inference(std::string prompt, FString* content, int time_per_action)
{
	std::ofstream("debug.txt", std::ios::app) << " Inference begin " << std::endl;
	FPerformanceDetector::GetInstance().ComputeStallTime();
	const int n_prompt = -llama_tokenize(model, prompt.c_str(), prompt.size(), NULL, 0, true, true);
	std::vector<llama_token> tokens_list(n_prompt);
	if (llama_tokenize(model, prompt.c_str(), prompt.size(), tokens_list.data(), tokens_list.size(), true, true) < 0) {
		fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
		return;
	}
	
	ctx_params.n_batch = n_prompt;
	ctx = llama_new_context_with_model(model, ctx_params);

	auto sparams = llama_sampler_chain_default_params();
	sparams.no_perf = false;
	
	llama_sampler * smpl = llama_sampler_chain_init(sparams);

	llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

	// 定义层数和对应的概率
	int layers[] = {0, 4, 5, 6, 7, 8, 10, 13};
	double probabilities[] = {1.0/372, 6.0/372, 36.0/372, 173.0/372, 132.0/372, 24.0/372};
	int num_layers = sizeof(layers) / sizeof(layers[0]);

	// 定义序列长度
	int sequence_length = 6 * 60000 / time_per_action; // 可以修改为任意长度

	// 生成序列
	int* sequence = generate_sequence(layers, probabilities, num_layers, sequence_length);
	
    // print the prompt token-by-token

	for (auto id : tokens_list) {
		char buf[128];
		int n = llama_token_to_piece(model, id, buf, sizeof(buf), 0, true);
		if (n < 0) {
			//fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
			return;
		}
		std::string s(buf, n);
		//printf("%s", s.c_str());
	}
	
	int action = 0;
	//FTaskDetector::GetInstance().SetTotalInferenceIsRunning(true);

	double BeginTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	while (true)
	{
		FTaskDetector::GetInstance().SetActionIsRunning(true);
		// create a llama_batch with size 512
		// we use this object to submit token data for decoding

		llama_batch batch = llama_batch_get_one(tokens_list.data(), tokens_list.size());
		std::ofstream("debug.txt", std::ios::app) << " New Action: " << action << std::endl;
		
		
		const auto t_main_start = ggml_time_us();
	
		int n_decode = 0;
		llama_token new_token_id;
		
	    // main loop

		double StallTimeForAction = 0;
		
		if (action >= 3 && CheckAutoUpdateSkipLayers())
		{
			StallTimeForAction = FPerformanceDetector::GetInstance().PredictNextStallTimeWindow();
			std::ofstream("debug.txt", std::ios::app) << " StallTimeForAction: " << StallTimeForAction << std::endl;
			FTaskScheduler::GetInstance().UpdateSkipDepthAfterPrefilling(StallTimeForAction, decode_time_per_layer, decode_time_per_token, prefill_time_per_layer, prefill_time_per_token);
		}
		int skip_layers = 0;

		FTaskDetector::GetInstance().SetIsPrefilling(true);
 
		std::string s = "new action: ";
		s += std::to_string(action);
		
		*content = UTF8_TO_TCHAR(s.c_str());

		int skip_layers = sequence[action % sequence_length];
		std::ofstream("debug.txt", std::ios::app) << " Skip Layers: " << skip_layers << std::endl;
		
		
		//std::string response = "";
		
	    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_len; ) {

	    	SCOPE_CYCLE_COUNTER(STAT_OneToken)

	    	FTaskDetector::GetInstance().SetIsInInferring(true);
	    	
    		if (action >= 3)
    		{
    			//skip_layers = FTaskScheduler::GetInstance().GetSkipDepth();
    			//skip_layers = sequence[action];
    			
    		}

    		if (llama_decode_skip_layers_v2(ctx, batch, skip_layers))
    		{
    			fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
    			return;
    		}
	    	
    		n_pos += batch.n_tokens;

    		// sample the next token
		    {
			    new_token_id = llama_sampler_sample(smpl, ctx, -1);

    			n_decode += 1;
    			
    			//std::string s(buf, n);

    			// is it an end of generation?
    			if (llama_token_is_eog(model, new_token_id)) {
    				break;
    			}

    			char buf[128];
    			int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
    			if (n < 0) {
    				//fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
    				return;
    			}

    			// prepare the next batch with the sampled token
    			batch = llama_batch_get_one(&new_token_id, 1);
		    }

	    	FTaskDetector::GetInstance().SetIsInInferring(false);
	    }

		FTaskDetector::GetInstance().SetActionIsRunning(false);
		
		const auto t_main_end = ggml_time_us();
		
		//llama_sampler_free(smpl);
		llama_kv_cache_clear(ctx);

		AvgInferenceTime = (t_main_end - t_main_start) / 1000.0f / n_decode;
		TotalInferenceTime = (t_main_end - t_main_start) / 1000.0f;

		std::ofstream("debug.txt", std::ios::app) << "Time per action: " << TotalInferenceTime << std::endl;
		
		if ((t_main_end - t_main_start) / 1000 < time_per_action)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(time_per_action - (t_main_end - t_main_start) / 1000));
		}

		action++;
		FTaskDetector::GetInstance().SetActionCount(action);
	}
}

float FNewLlamaModel::GetInferenceTime()
{
	return AvgInferenceTime;
}

float FNewLlamaModel::GetTotalInferenceTime()
{
	return TotalInferenceTime;
}

