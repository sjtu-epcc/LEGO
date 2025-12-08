#include "../Public/LlamaInferenceTask.h"


// void FLlamaInferenceTask::DoTask(ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent)
// {
// 	FLlamaModel::GetInstance().Inference(prompt, content);
// 	
// 	OnInferenceCompleted.ExecuteIfBound();
// }

// void FLlamaInferenceTask::DoWork()
// {
// 	FLlamaModel::GetInstance().Inference(prompt, content);
// }

bool FLlamaInferenceTask::Init()
{
	return true;
}

uint32_t FLlamaInferenceTask::Run()
{
	//FLlamaModel::GetInstance().Inference(prompt, content, AutoScheduler);
	FNewLlamaModel::GetInstance().Inference(prompt, content, TPA);
	
	return 0;
}

void FLlamaInferenceTask::Exit()
{
	(*order) += 1;
}








