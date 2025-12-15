#include "..\Public\TaskDetector.h"
#include "HAL/RunnableThread.h"
#include "Windows.h"

DECLARE_STATS_GROUP(TEXT("Rendering"), STATGROUP_LLAMA, STATCAT_Advanced);
DECLARE_CYCLE_STAT_EXTERN(TEXT("Rendering Task time"), STAT_Rendering, STATGROUP_LLAMA, );
DEFINE_STAT(STAT_Rendering);

uint32 FRenderTaskDetector::Run()
{
	while (true)
	{
		if (!FTaskDetector::GetInstance().GetRenderingTaskIsRunning())
		{
			SCOPE_CYCLE_COUNTER(STAT_Rendering);
			while (!FTaskDetector::GetInstance().GetRenderingTaskIsRunning());
		}
	}
	return 0;
}

FTaskDetector::FTaskDetector()
	: bRenderingTaskIsRunning(false)
	, bIsInFrameRendering(false)
	, bIsInInferring(false)
	, bRHITaskIsRunning(false)
	, WaitForRenderingCompletionTaskCount(0)
    , bTotalInferenceIsRunning(false)
	, bFirstDrawTask(true)
	, bIsInGraphComputing(false)
	, NodeCount(0)
	, ScheduledCount(0)
{
	Detector = new FRenderTaskDetector();
	//FRunnableThread::Create(Detector, TEXT("RenderingTask"), 0, TPri_Normal);
}

FTaskDetector& FTaskDetector::GetInstance()
{
	static FTaskDetector Instance;
	return Instance;
}

bool FTaskDetector::GetRenderingTaskIsRunning()
{
	return bRenderingTaskIsRunning;
}

TAtomic<bool>& FTaskDetector::GetIsInFrameRendering()
{
	return bIsInFrameRendering;
}

TAtomic<bool>& FTaskDetector::GetIsInInferring()
{
	return bIsInInferring;
}

TAtomic<bool>& FTaskDetector::GetRHITaskIsRunning()
{
	return bRHITaskIsRunning;
}

TAtomic<bool>& FTaskDetector::GetIsPrefilling()
{
	return bIsPrefilling;
}

TAtomic<bool>& FTaskDetector::GetActionIsRunning()
{
	return bActionIsRunning;
}

TAtomic<bool>& FTaskDetector::GetTotalInferenceIsRunning()
{
	return bTotalInferenceIsRunning;
}

TAtomic<bool>& FTaskDetector::GetFirstDrawTask()
{
	return bFirstDrawTask;
}

void FTaskDetector::SetFirstDrawTask(const bool State)
{
	bFirstDrawTask = State;
}


void FTaskDetector::SetTotalInferenceIsRunning(const bool State)
{
	bIsInInferring = State;
}



void FTaskDetector::SetRenderingTaskIsRunning(const bool State)
{
	bRenderingTaskIsRunning = State;
}

void FTaskDetector::SetIsInFrameRendering(const bool State)
{
	bIsInFrameRendering = State;
}

void FTaskDetector::SetIsInInferring(const bool State)
{
	bIsInInferring = State;
}

void FTaskDetector::SetRHITaskIsRunning(const bool State)
{
	bRHITaskIsRunning = State;
}

void FTaskDetector::SetIsPrefilling(const bool State)
{
	bIsPrefilling = State;
}


void FTaskDetector::SetActionIsRunning(const bool State)
{
	bRenderingTaskIsRunning = State;
}

void FTaskDetector::SetNodeCount(const int State)
{
	NodeCount = State;
}

TAtomic<int>& FTaskDetector::GetNodeCount()
{
	return NodeCount;
}

void FTaskDetector::SetScheduledCount(const int State)
{
	ScheduledCount = State;
}

TAtomic<int>& FTaskDetector::GetScheduledCount()
{
	return ScheduledCount;
}

TAtomic<bool>& FTaskDetector::GetIsInGraphComputing()
{
	return bIsInGraphComputing;
}

void FTaskDetector::SetIsInGraphComputing(const bool State)
{
	bIsInGraphComputing = State;
}

TAtomic<int>& FTaskDetector::GetActionCount()
{
 	return ActionCount; 
}

void FTaskDetector::SetActionCount(const int State)
{
	ActionCount = State;
}

 






























