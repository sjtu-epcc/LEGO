#pragma once

#include "LlamaModel\Public\Llama.h"
#include "NewLlamaModel/Public/NewLlama.h"


DECLARE_DELEGATE(FInferenceCompletedDelegate);

//class DLLEXPORT FLlamaInferenceTask : public FNonAbandonableTask
class DLLEXPORT FLlamaInferenceTask : public FRunnable
{
public:

	//friend class FAutoDeleteAsyncTask<FLlamaInferenceTask>;

	std::string prompt;
	
	FString* content;

	int32* order;

	int TPA;
	
	FLlamaInferenceTask(std::string prompt, FString* content, int32* order, int TPA)
		: prompt(prompt), content(content), order(order), TPA(TPA) {}

	//~FLlamaInferenceTask() { (*order) += 1; }
	FInferenceCompletedDelegate OnInferenceCompleted;
	
	//void DoTask(ENamedThreads::Type CurrentThread, const FGraphEventRef& MyCompletionGraphEvent);
	//void DoWork();
	
	// static ENamedThreads::Type GetDesiredThread() { return ENamedThreads::AnyThread; }
	// static ESubsequentsMode::Type GetSubsequentsMode() { return ESubsequentsMode::TrackSubsequents; }
	
	// FORCEINLINE TStatId GetStatId() const
	// {
	// 	RETURN_QUICK_DECLARE_CYCLE_STAT(FLlamaInferenceTask, STATGROUP_ThreadPoolAsyncTasks);
	// }
	virtual bool Init();
	virtual uint32_t Run();
	virtual void Exit();
};
