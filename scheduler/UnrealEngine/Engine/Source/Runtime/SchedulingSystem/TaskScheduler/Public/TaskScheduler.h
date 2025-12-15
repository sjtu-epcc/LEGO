#pragma once

#include "SchedulingSystem/DurationPredictor/Public/DurationPredictor.h"
#include "SchedulingSystem/TaskDetector/Public/TaskDetector.h"
#include "SchedulingSystem/PerformanceDetector/Public/PerformanceDetector.h"
#include "CoreMinimal.h"

class DLLEXPORT FTaskScheduler
{
public:
	static FTaskScheduler& GetInstance();
	void WaitForRenderingTaskCompletion();
	
	int GetExitLayerNum();
	int GetDecodePhase();
	int GetSkipDepth();
	void SetSkipDepth(const int SkipDepth);
	void SetDecodePhase(const int Phase);
	void SetSLOTime(const int Time);
	void SetTokenBeginTime(const double Time);
	void SetAllGraphNodes(const int Nodes);
	void SetExeGraphNodes(const int Nodes);
	void SetFirstLayerNum(const int Num);
	void SetNowLayerNum(const int Num);
	void SetIsRenderingEnd(const bool flag);
	void SetUpdateInPhase2(const bool flag);
	
	int GetSkipPolicy();
	void SetSkipPolicy(const int Policy);

	int GetOpPolicy();
	void SetOpPolicy(const int Policy);
	
	void UpdateSkipDepth1();
	void UpdateSkipDepth2();

	void UpdateSkipDepthAfterPrefilling(const double StallTimeForAction, const double DecodeTimePerLayer, const double DecodeTimePerToken, const double PrefillTimePerLayer, const double PrefillTimePerToken);
	
	~FTaskScheduler();
private:
	FTaskScheduler();
	TAtomic<int> Performance;
	TAtomic<int> ExitLayerNum;
	TAtomic<int> TokenDecodePhase;
	TAtomic<double> TokenBeginTime;
	TAtomic<double> TokenDecodingTime;
	TAtomic<bool> IsRenderingEnd; 
	TAtomic<int>  AllGraphNodes;
	TAtomic<int>  ExeGraphNodes;
	TAtomic<int> FirstLayerNum;
	int NowLayerNum;
	TAtomic<int> SLOTime;
	bool UpdateInPhase2;

	TAtomic<int> OpPolicy;
	TAtomic<int> SkipPolicy;
	TAtomic<int> SkipDepth;

	
};

class HighPrecisionTimer {
public:
	void SpinSleepMicroseconds(int64_t microseconds);
	static HighPrecisionTimer& GetInstance();
private:
	HighPrecisionTimer();
};


