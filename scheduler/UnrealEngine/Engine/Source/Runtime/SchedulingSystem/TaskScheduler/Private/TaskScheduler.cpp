#pragma once
#include "../Public/TaskScheduler.h"
#include "HAL/RunnableThread.h"

#include <chrono>
#include <msctf.h>

#include "Windows.h"

#define LastTwoLayersTime 1.5f
#define OneLayerNodes 32
#define OneLayerTime 0.4f
#define MaxExitLayerNum 29
#define MinExitLayerNum 13

HighPrecisionTimer::HighPrecisionTimer()
{
}

void HighPrecisionTimer::SpinSleepMicroseconds(int64_t microseconds)
{
	LARGE_INTEGER start, current, frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start);
        
	// 计算需要等待的计数值
	int64_t counts = (microseconds * frequency.QuadPart) / 1000000;
        
	do {
		QueryPerformanceCounter(&current);
	} while (current.QuadPart - start.QuadPart < counts);
}


HighPrecisionTimer& HighPrecisionTimer::GetInstance()
{
	static HighPrecisionTimer Instance;
	return Instance;
}

FTaskScheduler::FTaskScheduler()
	: Performance(0)
	, ExitLayerNum(MaxExitLayerNum)
	, TokenDecodePhase(0)
	, FirstLayerNum(0)
	, SLOTime(50)
	, UpdateInPhase2(true)
	, SkipPolicy(0)
	, SkipDepth(0)
	, OpPolicy(0)
{
}

FTaskScheduler::~FTaskScheduler()
{
}

FTaskScheduler& FTaskScheduler::GetInstance()
{
    static FTaskScheduler Instance;
    return Instance;
}

int FTaskScheduler::GetExitLayerNum()
{
	return ExitLayerNum;
}

int FTaskScheduler::GetSkipDepth()
{
	return SkipDepth;
}

int FTaskScheduler::GetDecodePhase()
{
	return TokenDecodePhase;
}

void FTaskScheduler::SetSkipDepth(const int nSkipDepth)
{
	SkipDepth = nSkipDepth;
}

void FTaskScheduler::SetDecodePhase(const int Phase)
{
	TokenDecodePhase = Phase;
}

void FTaskScheduler::SetSLOTime(const int Time)
{
	SLOTime = Time;
}

void FTaskScheduler::SetIsRenderingEnd(const bool flag)
{
	IsRenderingEnd = flag;
}

void FTaskScheduler::SetTokenBeginTime(const double Time)
{
	TokenBeginTime = Time;
}

void FTaskScheduler::SetAllGraphNodes(const int Nodes)
{
	AllGraphNodes = Nodes;
}

void FTaskScheduler::SetExeGraphNodes(const int Nodes)
{
	ExeGraphNodes = Nodes;
}

void FTaskScheduler::SetFirstLayerNum(const int Num)
{
	FirstLayerNum = Num;
}

void FTaskScheduler::SetNowLayerNum(const int Num)
{
	NowLayerNum = Num;
}

void FTaskScheduler::SetUpdateInPhase2(const bool flag)
{
	UpdateInPhase2 = flag;
}

int FTaskScheduler::GetSkipPolicy()
{
	return SkipPolicy;
}

void FTaskScheduler::SetSkipPolicy(const int nPolicy)
{
	SkipPolicy = nPolicy;
}

int FTaskScheduler::GetOpPolicy()
{
	return OpPolicy;
}

void FTaskScheduler::SetOpPolicy(const int Policy)
{
	OpPolicy = Policy;
}

void FTaskScheduler::UpdateSkipDepth1()
{
	if (SkipPolicy == 0 && OpPolicy == 0)
	{
		double NowTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		double TokenRemainTime = FMath::Max(SLOTime  - 3 - (NowTime - TokenBeginTime) / 1000.0, 0.0);
		double AllDecodeTime = FPerformanceDetector::GetInstance().GetRemainTimeSlice();
		double DeltaTime = TokenRemainTime - AllDecodeTime;
		double AvgRenderingTime = FPerformanceDetector::GetInstance().GetAvgRenderingTime();
		double DecodeTime = FMath::Max(FMath::Min(TokenRemainTime, AllDecodeTime), 0.0);
		if (DeltaTime <= AvgRenderingTime)
		{
	        if (DecodeTime <= LastTwoLayersTime)
			{
				ExitLayerNum = FMath::Max(NowLayerNum - 1, MinExitLayerNum);
        		//std::ofstream("debug.txt", std::ios::app) << "Phase: " << TokenDecodePhase << " case 1-1 TokenRemainTime: " << TokenRemainTime << " AllDecodeTime: " << AllDecodeTime << " DeltaTime: " << DeltaTime << " AvgRenderingTime: " << AvgRenderingTime << " ExitLayerNum: " << ExitLayerNum << "ExeNodes" << ExeGraphNodes << "AllNodes" << AllGraphNodes << "NowLayer: " << NowLayerNum << std::endl;
			}
			else
			{
				DecodeTime -= LastTwoLayersTime;
				ExitLayerNum = FMath::Max(
					int(DecodeTime/OneLayerTime + ExeGraphNodes*1.0f/OneLayerNodes + FirstLayerNum - 1), MinExitLayerNum);
				ExitLayerNum = FMath::Min(ExitLayerNum.Load(), MaxExitLayerNum);
				//std::ofstream("debug.txt", std::ios::app) << "Phase: " << TokenDecodePhase << " case 1-2 TokenRemainTime: " << TokenRemainTime << " AllDecodeTime: " << AllDecodeTime << " DeltaTime: " << DeltaTime << " AvgRenderingTime: " << AvgRenderingTime << " ExitLayerNum: " << ExitLayerNum << "ExeNodes" << ExeGraphNodes << "AllNodes" << AllGraphNodes << "NowLayer: " << NowLayerNum << std::endl;
			}
			UpdateInPhase2 = false;
		}
		//std::ofstream("debug.txt", std::ios::app) << "Phase: " << TokenDecodePhase << " case 1-3 TokenRemainTime: " << TokenRemainTime << " AllDecodeTime: " << AllDecodeTime << " DeltaTime: " << DeltaTime << " AvgRenderingTime: " << AvgRenderingTime << " ExitLayerNum: " << ExitLayerNum << "ExeNodes" << ExeGraphNodes << "AllNodes" << AllGraphNodes << "NowLayer: " << NowLayerNum << std::endl;
	}
}

void FTaskScheduler::UpdateSkipDepth2()
{
	if (SkipPolicy == 0 && OpPolicy == 0)
	{
		double NowTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		double TokenRemainTime = FMath::Max(SLOTime  - 3 - (NowTime - TokenBeginTime) / 1000.0, 0.0);
		double AllDecodeTime = FPerformanceDetector::GetInstance().GetRemainTimeSlice();
		//std::ofstream("debug.txt", std::ios::app) << "Phase: " << TokenDecodePhase << " case 2 TokenRemainTime: " << TokenRemainTime << " AllDecodeTime: " << AllDecodeTime << " ExitLayerNum: " << " ExeNodes: " << ExeGraphNodes << " AllNodes: " << AllGraphNodes << " NowLayer: " << NowLayerNum << std::endl;
		if (TokenDecodePhase == 0 && UpdateInPhase2)
		{
			if (IsRenderingEnd)
			{
				double DecodeTime = FMath::Max(FMath::Min(TokenRemainTime, AllDecodeTime), 0.0);
				ExitLayerNum = FMath::Min(NowLayerNum + int(DecodeTime / OneLayerTime), MaxExitLayerNum);
			}
			else
			{
				ExitLayerNum = MaxExitLayerNum;
			}
			//std::ofstream("debug.txt", std::ios::app) << "Phase: " << TokenDecodePhase << " case 2-1 TokenRemainTime: " << TokenRemainTime << " AllDecodeTime: " << AllDecodeTime << " ExitLayerNum: " << " ExeNodes: " << ExeGraphNodes << " AllNodes: " << AllGraphNodes << " NowLayer: " << NowLayerNum << std::endl;
		}
		else if (TokenDecodePhase == 2)
		{
			if (NowLayerNum <= MaxExitLayerNum)
			{
				UpdateSkipDepth1();
			}
			//std::ofstream("debug.txt", std::ios::app) << "Phase: " << TokenDecodePhase << " case 2-2 TokenRemainTime: " << TokenRemainTime << " AllDecodeTime: " << AllDecodeTime  << " ExitLayerNum: " << " ExeNodes: " << ExeGraphNodes << " AllNodes: " << AllGraphNodes << " NowLayer: " << NowLayerNum << std::endl;
		}
	}
}

void FTaskScheduler::UpdateSkipDepthAfterPrefilling(const double StallTimeForAction, const double DecodeTimePerLayer, const double DecodeTimePerToken, const double PrefillTimePerLayer, const double PrefillTimePerToken)
{
	double ActionTime = PrefillTimePerToken + 15*DecodeTimePerToken;
	double LayerTime = PrefillTimePerLayer + 15*DecodeTimePerLayer;
	if (StallTimeForAction >= ActionTime)
	{
		SkipDepth = 0;
	}
	else
	{
		SkipDepth = FMath::Max((int)std::ceil((ActionTime - StallTimeForAction - 20) / LayerTime),0);
		SkipDepth = SkipDepth.Load();
	}
	std::ofstream("debug.txt", std::ios::app) << " Skip Layers: " << SkipDepth << std::endl;
}


void FTaskScheduler::WaitForRenderingTaskCompletion()
{
	// OpPolicy:
	// = 0: 使用所有的空闲时间
	// = 1: Pilotfish
	// = 2: 自由竞争
	/*if (OpPolicy == 1)
	{
		while (true)
		{
			bool IsInFrameRendering = FTaskDetector::GetInstance().GetIsInFrameRendering();
			bool RenderingTaskIsRunning = FTaskDetector::GetInstance().GetRenderingTaskIsRunning();
			bool RHITaskIsRunning = FTaskDetector::GetInstance().GetRHITaskIsRunning();
			bool FirstDrawTask = FTaskDetector::GetInstance().GetFirstDrawTask();
			if (IsInFrameRendering/* && FPerformanceDetector::GetInstance().GetTargetAPM() == 300#1#)
			{
				if (FirstDrawTask) return;
			}
			else 
			{
				if (!IsInFrameRendering && !RHITaskIsRunning && !RenderingTaskIsRunning)
				{
					/*if (OpPolicy.Load() == 2)
					{
						while (!FTaskDetector::GetInstance().GetRHITaskIsRunning()) {}
						return;
					}#1#
					if (FPerformanceDetector::GetInstance().GetTimeSlice() != 0)
					{
						return;
					}
				}
			}
		}
	}*/
	
	if (OpPolicy == 2) return;

	else
	{
		while (true)
		{
			bool IsInFrameRendering = FTaskDetector::GetInstance().GetIsInFrameRendering();
			bool RenderingTaskIsRunning = FTaskDetector::GetInstance().GetRenderingTaskIsRunning();
			bool RHITaskIsRunning = FTaskDetector::GetInstance().GetRHITaskIsRunning();

			if (IsInFrameRendering)
			{
				/*if (OpPolicy.Load() == 2)
				{
					return;
				}*/
			
				if (/*!RHITaskIsRunning &&*/ !RenderingTaskIsRunning && OpPolicy == 0)
				{
					return;
				}
			}
			else
			{
					if (!RHITaskIsRunning && !RenderingTaskIsRunning)
					{
						/*if (OpPolicy.Load() == 2)
						{
							while (!FTaskDetector::GetInstance().GetRHITaskIsRunning()) {}
							return;
						}*/
						if (FPerformanceDetector::GetInstance().GetTimeSlice() != 0)
						{
							return;
						}
					}
				
				/*else
				{
					if (OpPolicy.Load() == 2)
					{
						return;
					}
				}*/
			}
		}	
	}
	
}



















