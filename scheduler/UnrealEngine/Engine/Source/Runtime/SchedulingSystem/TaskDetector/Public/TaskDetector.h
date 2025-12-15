#pragma once
#include <mutex>


class FRenderTaskDetector: public FRunnable
{
public:
	
	virtual bool Init() { return true; }
	virtual uint32_t Run();
	virtual void Exit() {}
};

class DLLEXPORT FTaskDetector
{
public:
	FRenderTaskDetector* Detector;
	int WaitForRenderingCompletionTaskCount;
	std::mutex MutexForWaitForRenderingCompletionTaskCount;
	static FTaskDetector& GetInstance();
	bool GetRenderingTaskIsRunning();
	TAtomic<bool>& GetIsInFrameRendering();
	TAtomic<bool>& GetRHITaskIsRunning();
	TAtomic<bool>& GetIsInInferring();
	TAtomic<bool>& GetIsPrefilling();
	TAtomic<bool>& GetActionIsRunning();
	TAtomic<bool>& GetTotalInferenceIsRunning();
	TAtomic<bool>& GetFirstDrawTask();
	TAtomic<bool>& GetIsInGraphComputing();
	TAtomic<int>& GetNodeCount();
	TAtomic<int>& GetScheduledCount();
	TAtomic<int>& GetActionCount();
	void SetRenderingTaskIsRunning(const bool State);
	void SetIsInFrameRendering(const bool State);
	void SetIsInInferring(const bool State);
	void SetRHITaskIsRunning(const bool State);
	void SetIsPrefilling(const bool State);
	void SetActionIsRunning(const bool State);
	void SetTotalInferenceIsRunning(const bool State);
	void SetFirstDrawTask(const bool State);
	void SetNodeCount(const int State);
	void SetScheduledCount(const int State);
	void SetIsInGraphComputing(const bool State);
	void SetActionCount(const int State);
private:
	FTaskDetector();
	TAtomic<bool> bRHITaskIsRunning;
	TAtomic<bool> bRenderingTaskIsRunning;
	TAtomic<bool> bIsInFrameRendering;
	TAtomic<bool> bIsInInferring;
	TAtomic<bool> bIsPrefilling;
	TAtomic<bool> bActionIsRunning;
	TAtomic<bool> bTotalInferenceIsRunning;
	TAtomic<bool> bFirstDrawTask;
	TAtomic<bool> bIsInGraphComputing;
	TAtomic<int> NodeCount;
	TAtomic<int> ScheduledCount;
	TAtomic<int> ActionCount;
};
