#include "..\Public\Llama.h"
#include "llamacpp/ggml-cuda.h"

std::string llama_token_to_piece(llama_context * ctx, llama_token token) {
	std::vector<char> result(8, 0);
	const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
	if (n_tokens < 0) {
		result.resize(-n_tokens);
		int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
		GGML_ASSERT(check == -n_tokens);
	}
	else {
		result.resize(n_tokens);
	}

	return std::string(result.data(), result.size());
}

FLlamaModel& FLlamaModel::GetInstance()
{
	static FLlamaModel llama;
	return llama;
}


void FLlamaModel::Init()
{
	register_ggml_cuda_compute_forward(ggml_cuda_compute_forward_scheduled);
	
	params.model = ".\\llama-2-7b-chat.Q8_0.gguf";
	
	n_len = 600;
	
	llama_backend_init(params.numa);
	
	model_params = llama_model_default_params();
	model_params.n_gpu_layers = 10000;
	model = llama_load_model_from_file(params.model.c_str(), model_params);
	
	ctx_params = llama_context_default_params();
	ctx_params.seed = time(nullptr);
	ctx_params.n_ctx = 2048;
	ctx_params.n_threads = params.n_threads;
	ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;
	ctx = llama_new_context_with_model(model, ctx_params);
}

void FLlamaModel::Exit()
{
	llama_free(ctx);
	llama_free_model(model);
	llama_backend_free();

	fprintf(stderr, "Llama has exited.\n");
}


void FLlamaModel::Inference(std::string input_prompt, FString* content, bool AutoSchedule)
{
	FTaskDetector::GetInstance().SetIsInInferring(true);
	params.prompt = input_prompt;
	std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, params.prompt, true);

    const int n_ctx    = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    //printf("\n%s: n_len = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_len, n_ctx, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        printf("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        printf("%s:        either reduce n_len or increase n_ctx\n", __func__);
        return;
    }
	
    // print the prompt token-by-token

    fprintf(stderr, "\n");

    for (auto id : tokens_list) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

	

    if (llama_decode(ctx, batch) != 0) {
        printf("%s: llama_decode() failed\n", __func__);
        return;
    }

    // main loop

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();
	double start_time = FDateTime::Now().GetTimeOfDay().GetTotalSeconds();

	std::string response = "";

	bool spaceflag = false;
	
    while (n_cur <= n_len) {
        // sample the next token
        {
            auto   n_vocab = llama_n_vocab(model);
            auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // sample the most likely token
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);
        	
            // is it an end of stream?
            if (new_token_id == llama_token_eos(model) || n_cur == n_len) {
                printf("\n");

                break;
            }
        		
			std::string token = llama_token_to_piece(ctx, new_token_id);
        	
        	if (spaceflag == false && (token[0] != '\n' && token[0] != ' ' && token[0] != '\t'))
        			spaceflag = true;
        	if (spaceflag == true) response += token.c_str();
        	
        	*content = UTF8_TO_TCHAR(response.c_str());
        	
            //printf("%s", token.c_str());
            //fflush(stdout);

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);
        	
        }

        n_cur += 1;
    	
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        }

    	n_decode += 1;
    }
	FTaskDetector::GetInstance().SetIsInInferring(false);
	

	llama_batch_free(batch);
	llama_kv_cache_clear(ctx);

	printf("\n");

	const auto t_main_end = ggml_time_us();

	printf("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
			__func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

	llama_print_timings(ctx);

	InferenceSpeed = n_decode / ((t_main_end - t_main_start) / 1000000.0f);

	fprintf(stderr, "\n");
}

float FLlamaModel::GetInferenceSpeed()
{
	return InferenceSpeed;
}

