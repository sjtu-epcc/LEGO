#include "PerformanceDetector.h"
#include <fstream>
#include <chrono>
#include <string>

#include "TaskDetector.h"
#include "TaskScheduler.h"

FPerformanceDetector::FPerformanceDetector()
	: FrameBeginTime(0)
	, FrameRenderingEndTime(0)
	, FrameTime(0)
	, TimeSlice(0)
	, AvgRenderingTime(0)
	, MaxFPS(60)
	, FrameStallTimeInRendering(0)
	, FrameStallTimeInferenceUnused(0)
	, FrameStallTimeInferenceUsed(0)
	, AllStallTimePrefillingUsed(0)
	, FrameCount(0)
	, StallTimeWindowTime(0)
	, TargetAPM(200)
{
	MonitorStallTimeInferenceUsedTask = new FMonitorStallTimeInferenceUsedTask();
	MonitorStallTimeInferenceUnusedTask = new FMonitorStallTimeInferenceUnusedTask();
	MonitorStallTimeInRenderingTask = new FMonitorStallTimeInRenderingTask();
	MonitorStallTimePrefillingUsedTask = new FMonitorStallTimePrefillingUsedTask();
	MonitorStallTimeAfterRenderingTask = new FMonitorStallTimeAfterRenderingTask();
}

FPerformanceDetector& FPerformanceDetector::GetInstance()
{
	static FPerformanceDetector Instance;
	return Instance;
}

float FPerformanceDetector::GetAverageFPS()
{
	extern ENGINE_API float GAverageFPS;
	return GAverageFPS;
}


double FPerformanceDetector::GetFrameBeginTime()
{
	return FrameBeginTime;
}

void FPerformanceDetector::ComputeTimeSlice()
{
	while (true)
	{
		if (!FTaskDetector::GetInstance().GetIsInFrameRendering() && !FTaskDetector::GetInstance().GetRenderingTaskIsRunning() && !FTaskDetector::GetInstance().GetRHITaskIsRunning())
		{
			if (FrameTime == 0)
			{
				FrameRenderingEndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				FrameTime = (FrameRenderingEndTime - FrameBeginTime) / 1000.0f;
				if (AvgRenderingTime == 0) AvgRenderingTime = FrameTime.Load();
				else AvgRenderingTime = 0.75 * AvgRenderingTime + 0.25 * FrameTime;
				const float nTimeSlice = 1000.0/MaxFPS - FrameTime;
				TimeSlice = FMath::Max(nTimeSlice, 0.0f);
				UpdateStallTimeWindowTime();
			}
		}
	}

}

double FPerformanceDetector::GetTimeSlice()
{
	return TimeSlice;
}

double FPerformanceDetector::GetRemainTimeSlice()
{
	double NowTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	return FMath::Max(TimeSlice - (NowTime - FrameRenderingEndTime) / 1000.0f, 0.0);
}

double FPerformanceDetector::GetAvgRenderingTime()
{
	return AvgRenderingTime;
}

void FPerformanceDetector::SetFrameBeginTime(const double Time)
{
	FrameBeginTime = Time;
}

void FPerformanceDetector::SetFrameTime(const double Time)
{
	FrameTime = Time;
}

double FPerformanceDetector::GetFrameTime()
{
	return FrameTime;
}

void FPerformanceDetector::SetMaxFPS(const double FPS)
{
	MaxFPS = FPS;
}

void FPerformanceDetector::WriteStallTimeInfo()
{
	/*if (FrameTime == 0)
	{
		FrameRenderingEndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		FrameTime = (FrameRenderingEndTime - FrameBeginTime) / 1000.0f;
		if (AvgRenderingTime == 0) AvgRenderingTime = FrameTime.Load();
		else AvgRenderingTime = 0.75 * AvgRenderingTime + 0.25 * FrameTime;
		const float nTimeSlice = 1000.0/MaxFPS - FrameTime;
		TimeSlice = FMath::Max(nTimeSlice, 0.0f);
		UpdateStallTimeWindowTime();
	}*/
	double FPS = GetAverageFPS();
	if (FrameTime > 1000.0 / 60) FPS = 1000.0 / FrameTime;
	if (FPS <= 61 /*&& FrameTime*/) 
	{
		if (OpPolicy != 1)
		{
			std::ofstream("debug.txt", std::ios::app)
				<< "FrameRenderingTime: " << FrameTime
				<< " TimeSlice: " << TimeSlice
				<< " StallTimeInRendering: " << FrameStallTimeInRendering
				//<< " StallTimeInferenceUnused: " << FrameStallTimeInferenceUnused
				<< " StallTimeInferenceUsed: " << TimeSlice + FrameStallTimeInRendering - FrameStallTimeInferenceUnused
				<< " FPS: " << FPS
				<< std::endl;
		}

		else
		{
			std::ofstream("debug.txt", std::ios::app)
				<< "FrameRenderingTime: " << FrameTime
				<< " TimeSlice: " << TimeSlice
				<< " StallTimeInRendering: " << FrameStallTimeInRendering
				//<< " StallTimeInferenceUnused: " << FrameStallTimeInferenceUnused
				<< " StallTimeInferenceUsed: " << TimeSlice - FrameStallTimeInferenceUnused
				<< " FPS: " << FPS
				<< std::endl;
		}

		/*std::string nums;
		for (int i = 0; i < FrameStallTimeInRenderingArray.size(); i++)
		{
			nums += std::to_string(FrameStallTimeInRenderingArray[i]);
			nums += " ";
		}
		std::ofstream("debug.txt", std::ios::app) << "FrameStallTimeInRendering: " << nums << std::endl;*/
	}
}

#define MAX_FRAME_COUNT 50

void FPerformanceDetector::UpdateStallTimeWindowTime()
{
	StallTimeWindowsQueue.push_back(TimeSlice);
	//StallTimeWindowsQueue.push_back(TimeSlice+FrameStallTimeInRendering);
	if (StallTimeWindowsQueue.size() > MAX_FRAME_COUNT)
		StallTimeWindowsQueue.pop_front();
	/*std::string nums;
	for (int i = 0; i < StallTimeWindowsQueue.size(); i++)
	{
		nums += std::to_string(StallTimeWindowsQueue[i]);
		nums += " ";
	}
	std::ofstream("debug.txt", std::ios::app) << "StallTimeWindowsQueue: " << nums << std::endl;*/
}

void FPerformanceDetector::SetWindowsCount(const int Count)
{
	WindowsCount = Count;
}

void FPerformanceDetector::SetWindowsLength(const int Length)
{
	WindowsLength = Length;
}

int FPerformanceDetector::GetWindowsLength()
{
	return WindowsLength;
}

int FPerformanceDetector::GetTargetAPM()
{
	return TargetAPM;
}

void FPerformanceDetector::SetTargetAPM(const int APM)
{
	TargetAPM = APM;
}


void FPerformanceDetector::UpdateFrameStallTime()
{
	FrameStallTimeInferenceUsed = 0;
	FrameStallTimeInferenceUnused = 0;
	FrameStallTimeInRendering = 0;
	FrameStallTimeInRenderingArray.clear();
}

double FPerformanceDetector::UpdateAllStallTimePrefillingUsed()
{
	std::ofstream("debug.txt", std::ios::app) << "StallTimePrefillingUsed: " << AllStallTimePrefillingUsed << std::endl;
	double res = AllStallTimePrefillingUsed;
	AllStallTimePrefillingUsed = 0;
	return res;
}

double FPerformanceDetector::PredictNextStallTimeWindow()
{
	//rural
	//std::vector<double> Weights = {1.469337, -0.28733, -0.22863, 8.787756}; //注意权重倒序存放, weights={w3,w2,w1,d} scene3-200apm
	//std::vector<double> Weights = {1.404611, -0.20713, -0.24817, 6.368396}; //注意权重倒序存放, weights={w3,w2,w1,d} scene3-300apm

	//std::vector<double> Weights = {1.108417, -0.11725, -0.0443, 8.51669}; //注意权重倒序存放, weights={w3,w2,w1,d} scene1-200apm
	//std::vector<double> Weights = {1.171115, -0.07491, -0.19473, 10.52871}; //注意权重倒序存放, weights={w3,w2,w1,d} scene1-300apm
	
	//std::vector<double> Weights = {1.451419, -0.24936, -0.25198, 8.69996}; //注意权重倒序存放, weights={w3,w2,w1,d} scene2-200apm
	//std::vector<double> Weights = {1.416931, -0.22387, -0.25057, 6.681748}; //注意权重倒序存放, weights={w3,w2,w1,d} scene2-300apm


	//watermills
	//std::vector<double> Weights = {0.928304, 0.00629, -0.08938, 18.14008}; //注意权重倒序存放, weights={w3,w2,w1,d} scene3-200apm
	std::vector<double> Weights = {0.873371, 0.119535, -0.05704, 11.27746}; //注意权重倒序存放, weights={w3,w2,w1,d} scene3-300apm

	//std::vector<double> Weights = {0.910026, 0.073817, -0.03962, 8.971267}; //注意权重倒序存放, weights={w3,w2,w1,d} scene2-200apm
	//std::vector<double> Weights = {0.946492, 0.002635, -0.07186, 13.15483}; //注意权重倒序存放, weights={w3,w2,w1,d} scene2-300apm
	
	double Res = 0, Tmp = 0;
	int Length = StallTimeWindowsQueue.size();
	for (int i = WindowsCount - 1; i >= 0; i--)
	{
		Tmp = 0;
		for (int j = WindowsLength; j > 0; j--)
		{
			Tmp += StallTimeWindowsQueue[Length - i - j];
		}
		Res += Tmp * Weights[i];
	}
	Res += Weights[WindowsCount];
	return Res;
}

void FPerformanceDetector::ComputeStallTimeInferenceUnused()
{
	double BeginTime = 0;
	while (true)
	{
		const bool IsInFrameRendering = FTaskDetector::GetInstance().GetIsInFrameRendering();
		const bool IsInInference = FTaskDetector::GetInstance().GetIsInInferring();
		const bool RenderingTaskIsRunning = FTaskDetector::GetInstance().GetRenderingTaskIsRunning();
		const bool RHITaskIsRunning = FTaskDetector::GetInstance().GetRHITaskIsRunning();
		if (IsInFrameRendering && !RenderingTaskIsRunning && !IsInInference && !RHITaskIsRunning)
		{
			if (BeginTime == 0)
			{
				BeginTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				while (FTaskDetector::GetInstance().GetIsInFrameRendering() && !FTaskDetector::GetInstance().GetIsInInferring() && !FTaskDetector::GetInstance().GetRenderingTaskIsRunning()) {}
				double EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				FrameStallTimeInferenceUnused = FrameStallTimeInferenceUnused + (EndTime - BeginTime) * 1.0f / 1000.0f;
				if (FTaskDetector::GetInstance().GetIsPrefilling()) AllStallTimePrefillingUsed = AllStallTimePrefillingUsed + (EndTime - BeginTime) * 1.0f / 1000.0f;
				BeginTime = 0;
			}
		}
	
		if (!IsInFrameRendering && !RenderingTaskIsRunning && !RHITaskIsRunning && !IsInInference)
		{
			if (BeginTime == 0)
			{
				BeginTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				while (!FTaskDetector::GetInstance().GetRHITaskIsRunning() && !FTaskDetector::GetInstance().GetIsInInferring()) {}
				double EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				FrameStallTimeInferenceUnused = FrameStallTimeInferenceUnused + (EndTime - BeginTime) * 1.0f / 1000.0f;
				if (FTaskDetector::GetInstance().GetIsPrefilling()) AllStallTimePrefillingUsed = AllStallTimePrefillingUsed + (EndTime - BeginTime) * 1.0f / 1000.0f;
				BeginTime = 0;
			}
		}
	}
}

void FPerformanceDetector::ComputeStallTimeInferenceUsed()
{
	double BeginTime = 0;
	while (true)
	{
		const bool IsInFrameRendering = FTaskDetector::GetInstance().GetIsInFrameRendering();
		const bool IsInInference = FTaskDetector::GetInstance().GetIsInInferring();
		const bool RenderingTaskIsRunning = FTaskDetector::GetInstance().GetRenderingTaskIsRunning();
		const bool RHITaskIsRunning = FTaskDetector::GetInstance().GetRHITaskIsRunning();
		if (IsInFrameRendering && !RenderingTaskIsRunning && IsInInference)
		{
			if (BeginTime == 0)
			{
				BeginTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				while (FTaskDetector::GetInstance().GetIsInFrameRendering() && FTaskDetector::GetInstance().GetIsInInferring() && !FTaskDetector::GetInstance().GetRenderingTaskIsRunning()) {}
				double EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				FrameStallTimeInferenceUsed = FrameStallTimeInferenceUsed + (EndTime - BeginTime) * 1.0f / 1000.0f;
				BeginTime = 0;
			}
		}
	
		if (!IsInFrameRendering && !RenderingTaskIsRunning && !RHITaskIsRunning && IsInInference)
		{
			if (BeginTime == 0)
			{
				BeginTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				while (!FTaskDetector::GetInstance().GetRHITaskIsRunning() && FTaskDetector::GetInstance().GetIsInInferring()) {}
				double EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				FrameStallTimeInferenceUsed = FrameStallTimeInferenceUsed + (EndTime - BeginTime) * 1.0f / 1000.0f;
				BeginTime = 0;
			}
		}
	}
}

void FPerformanceDetector::ComputeStallTimeInRendering()
{
	double BeginTime = 0;
	while (true)
	{
		const bool IsInFrameRendering = FTaskDetector::GetInstance().GetIsInFrameRendering();
		const bool RenderingTaskIsRunning = FTaskDetector::GetInstance().GetRenderingTaskIsRunning();
		if (IsInFrameRendering && !RenderingTaskIsRunning)
		{
			if (BeginTime == 0)
			{
				BeginTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				while (FTaskDetector::GetInstance().GetIsInFrameRendering() && !FTaskDetector::GetInstance().GetRenderingTaskIsRunning()) {}
				double EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				double NowTimeSlice = (EndTime - BeginTime) * 1.0f / 1000.0f;
				FrameStallTimeInRendering = FrameStallTimeInRendering + NowTimeSlice;
				FrameStallTimeInRenderingArray.push_back(NowTimeSlice);
				BeginTime = 0;
			}
		}
	}
}

void FPerformanceDetector::ComputeStallTimePrefillingUsed()
{
	double BeginTime = 0;
	while (true)
	{
		const bool IsInFrameRendering = FTaskDetector::GetInstance().GetIsInFrameRendering();
		const bool RenderingTaskIsRunning = FTaskDetector::GetInstance().GetRenderingTaskIsRunning();
		const bool RHITaskIsRunning = FTaskDetector::GetInstance().GetRHITaskIsRunning();
		if (IsInFrameRendering)
		{
			if (!RenderingTaskIsRunning && FTaskDetector::GetInstance().GetIsPrefilling())
			{
				if (BeginTime == 0)
				{
					BeginTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
					while (!FTaskDetector::GetInstance().GetRenderingTaskIsRunning() && FTaskDetector::GetInstance().GetIsPrefilling() && FTaskDetector::GetInstance().GetIsInFrameRendering()) {}
					double EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
					double NowTimeSlice = (EndTime - BeginTime) * 1.0f / 1000.0f;
					//std::ofstream("debug.txt", std::ios::app) << "StallTimePrefillingUsedSlice: " << NowTimeSlice << std::endl;
					AllStallTimePrefillingUsed = AllStallTimePrefillingUsed + NowTimeSlice;
					BeginTime = 0;
				}
			}
		}
		else
		{
			if (!RenderingTaskIsRunning && !RHITaskIsRunning && FTaskDetector::GetInstance().GetIsPrefilling())
			{
				if (BeginTime == 0)
				{
					BeginTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
					while (!FTaskDetector::GetInstance().GetRHITaskIsRunning() && FTaskDetector::GetInstance().GetIsPrefilling() && !FTaskDetector::GetInstance().GetIsInFrameRendering()) {}
					double EndTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
					double NowTimeSlice = (EndTime - BeginTime) * 1.0f / 1000.0f;
					//std::ofstream("debug.txt", std::ios::app) << "StallTimePrefillingUsedSlice: " << NowTimeSlice << std::endl;
					AllStallTimePrefillingUsed = AllStallTimePrefillingUsed + NowTimeSlice;
					BeginTime = 0;
				}
			}
		}
	}
}



void FPerformanceDetector::SetOpPolicy(int Policy)
{
	OpPolicy = Policy;
}

bool FMonitorStallTimeInferenceUsedTask::Init()
{
	return true;
}

uint32_t FMonitorStallTimeInferenceUsedTask::Run()
{
	FPerformanceDetector::GetInstance().ComputeStallTimeInferenceUsed();
	return 0;
}

void FMonitorStallTimeInferenceUsedTask::Exit()
{
	return;
}

bool FMonitorStallTimeInferenceUnusedTask::Init()
{
	return true;
}

uint32_t FMonitorStallTimeInferenceUnusedTask::Run()
{
	FPerformanceDetector::GetInstance().ComputeStallTimeInferenceUnused();
	return 0;
}

void FMonitorStallTimeInferenceUnusedTask::Exit()
{
	return;
}

bool FMonitorStallTimeInRenderingTask::Init()
{
	return true;
}

uint32_t FMonitorStallTimeInRenderingTask::Run()
{
	FPerformanceDetector::GetInstance().ComputeStallTimeInRendering();
	return 0;
}

void FMonitorStallTimeInRenderingTask::Exit()
{
	return;
}

bool FMonitorStallTimePrefillingUsedTask::Init()
{
	return true;
}

uint32_t FMonitorStallTimePrefillingUsedTask::Run()
{
	FPerformanceDetector::GetInstance().ComputeStallTimePrefillingUsed();
	return 0;
}

void FMonitorStallTimePrefillingUsedTask::Exit()
{
	return;
}

bool FMonitorStallTimeAfterRenderingTask::Init()
{
	return true;
}

uint32_t FMonitorStallTimeAfterRenderingTask::Run()
{
	FPerformanceDetector::GetInstance().ComputeTimeSlice();
	return 0;
}

void FMonitorStallTimeAfterRenderingTask::Exit()
{
	return;
}


void FPerformanceDetector::ComputeStallTime()
{
	FRunnableThread::Create(MonitorStallTimeInferenceUsedTask, TEXT("MonitorStallTimeForInferenceTask"), 0, TPri_Normal);
	FRunnableThread::Create(MonitorStallTimeInferenceUnusedTask, TEXT("MonitorStallTimeForInferenceTask"), 0, TPri_Normal);
	FRunnableThread::Create(MonitorStallTimeInRenderingTask, TEXT("MonitorStallTimeInRenderingTask"), 0, TPri_Normal);
	FRunnableThread::Create(MonitorStallTimePrefillingUsedTask, TEXT("MonitorStallTimePrefillingUsedTask"), 0, TPri_Normal);
	FRunnableThread::Create(MonitorStallTimeAfterRenderingTask, TEXT("MonitorStallTimeAfterRenderingTask"), 0, TPri_Normal);
}


















