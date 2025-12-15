#pragma once

#include "CoreMinimal.h"
#include <iostream>
#include <deque>
#include <vector>

class FMonitorStallTimeAfterRenderingTask : public FRunnable
{
public:
	
	FMonitorStallTimeAfterRenderingTask(){}

	virtual bool Init();
	virtual uint32_t Run();
	virtual void Exit();
};

class FMonitorStallTimeInRenderingTask : public FRunnable
{
public:
	
	FMonitorStallTimeInRenderingTask(){}

	virtual bool Init();
	virtual uint32_t Run();
	virtual void Exit();
};

class FMonitorStallTimeInferenceUnusedTask : public FRunnable
{
public:
	
	FMonitorStallTimeInferenceUnusedTask(){}

	virtual bool Init();
	virtual uint32_t Run();
	virtual void Exit();
};

class FMonitorStallTimeInferenceUsedTask : public FRunnable
{
public:
	
	FMonitorStallTimeInferenceUsedTask(){}

	virtual bool Init();
	virtual uint32_t Run();
	virtual void Exit();
};

class FMonitorStallTimePrefillingUsedTask : public FRunnable
{
public:
	
	FMonitorStallTimePrefillingUsedTask(){}

	virtual bool Init();
	virtual uint32_t Run();
	virtual void Exit();
};


class DLLEXPORT FPerformanceDetector
{
public:
	float GetAverageFPS();
	double GetFrameBeginTime();
	void ComputeTimeSlice();
	double GetTimeSlice();
	int GetWindowsLength();

	static FPerformanceDetector& GetInstance();

	void SetFrameBeginTime(const double Time);
	void SetFrameTime(const double Time);
	void SetMaxFPS(const double FPS);

	void WriteStallTimeInfo();
	double GetFrameTime();
	double GetRemainTimeSlice();
	double GetAvgRenderingTime();

	void SetWindowsCount(const int Count);
	void SetWindowsLength(const int Length);
	void UpdateStallTimeWindowTime();

	void ComputeStallTimeInRendering();
	void ComputeStallTimeInferenceUsed();
	void ComputeStallTimeInferenceUnused();
	void ComputeStallTimePrefillingUsed();
	void ComputeStallTime();

	void UpdateFrameStallTime();
	double UpdateAllStallTimePrefillingUsed();

	double PredictNextStallTimeWindow();

	void SetOpPolicy(int Policy);

	int GetTargetAPM();
	void SetTargetAPM(const int APM);
	
	
private:
	FPerformanceDetector();
	double FrameBeginTime;
	double FrameRenderingEndTime;
	double MaxFPS;
	TAtomic<int> FrameCount;
	TAtomic<float> FrameTime;
	TAtomic<double> TimeSlice;
	TAtomic<double> AvgRenderingTime;
	TAtomic<int> OpPolicy;

	double AllStallTimePrefillingUsed;
	double FrameStallTimeInRendering;
	double FrameStallTimeInferenceUnused;
	double FrameStallTimeInferenceUsed;

	std::deque<double> StallTimeWindowsQueue;
	int WindowsCount;
	int WindowsLength;
	double StallTimeWindowTime;

	int TargetAPM;

	std::vector<double> FrameStallTimeInRenderingArray;

	FMonitorStallTimeInRenderingTask* MonitorStallTimeInRenderingTask;
	FMonitorStallTimeInferenceUsedTask* MonitorStallTimeInferenceUsedTask;
	FMonitorStallTimeInferenceUnusedTask* MonitorStallTimeInferenceUnusedTask;
	FMonitorStallTimePrefillingUsedTask* MonitorStallTimePrefillingUsedTask;
	FMonitorStallTimeAfterRenderingTask* MonitorStallTimeAfterRenderingTask;
};