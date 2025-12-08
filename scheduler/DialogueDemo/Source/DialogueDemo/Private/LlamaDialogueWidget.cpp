// Fill out your copyright notice in the Description page of Project Settings.

#include "../Public/LlamaDialogueWidget.h"

ULlamaDialogueWidget::ULlamaDialogueWidget(const FObjectInitializer& ObjectInitialize)
	:Super(ObjectInitialize)
{
	Order = 0;
	Content = TEXT("What can I do for you?");
	prompts[0] = TEXT("This is a combat scenario. The objective is to reduce the enemy’s health to zero while maintaining your own health. The battle takes place in a three-dimensional space defined by X, Y, and Z coordinates. You are currently in a place labeled X, with dimensions X as 1, dimension Y as 2 and dimension Z as 3. You possess four skills: Skill A1, Skill B1, Skill C1, and Skill D1. Skill A1 is XXX. Skill B1 is XXX. Skill C1 is XXX. Skill D1 is XXX. The enemy possesses six skills: Skill A2, Skill B2, Skill C2, Skill D2, Skill E2, Skill F2. Skill A2 is XXX. Skill B2 is XXX. Skill C2 is XXX. Skill D2 is XXX. Skill E2 is XXX. Skill F2 is XXX.Your current health is XXX, and your position is (X, Y, Z). Your next possible positions are (X+1, Y+1, Z+1), (X-1, Y+1, Z+1), (X+1, Y-1, Z+1). Your available skills are A1, B1, C1. Your enemy's health is XXX, and his position is (X, Y, Z). His next possible positions are (X+1, Y+1, Z+1), (X-1, Y+1, Z+1), (X+1, Y-1, Z+1). His available skills are A2, B2, C2. Your last action involved moving from position A to position B and using Skill C. However, the effect of this skill was negative. Conversely, your enemy’s last move was from position C to position D, where they utilized Skill D, resulting in a positive effect. The stage is set for a strategic battle where every move and skill choice will influence the outcome.");
	Prompt = "";
	AutoSchedule = true;
	APM = 200;
}

bool ULlamaDialogueWidget::GetNextPrompt()
{
	if (Order < 2)
	{
		Content = prompts[Order / 2];
		Prompt = std::string(TCHAR_TO_UTF8(*Content));
		Order += 1;
		
		return true;
	}
	return false; 
}

void ULlamaDialogueWidget::LlamaInference()
{
	double MaxFPS = IConsoleManager::Get().FindConsoleVariable(TEXT("t.MaxFPS"))->GetFloat();
	FPerformanceDetector::GetInstance().SetMaxFPS(MaxFPS);
	//std::string Prompt = TCHAR_TO_UTF8(*Content);
	Content = TEXT("generating...");

	double TPA = 60.0 * 1000 / APM;
	int frame_per_action = TPA / (1000 / MaxFPS);
	FPerformanceDetector::GetInstance().SetWindowsLength(frame_per_action);
	FPerformanceDetector::GetInstance().SetWindowsCount(3);
	FLlamaInferenceTask* InferenceTask = new FLlamaInferenceTask(Prompt, &Content, &Order, TPA);
	FRunnableThread::Create(InferenceTask, TEXT("InferenceTask"), 0, TPri_Normal);
}

float ULlamaDialogueWidget::GetAverageFPS()
{
	extern ENGINE_API float GAverageFPS;
	return GAverageFPS;
}

float ULlamaDialogueWidget::GetInferenceTime()
{
	return FNewLlamaModel::GetInstance().GetInferenceTime();
}

float ULlamaDialogueWidget::GetTotalInferenceTime()
{
	return FNewLlamaModel::GetInstance().GetTotalInferenceTime();
}


void ULlamaDialogueWidget::SetSkipDepth(const int SkipDepth)
{
	FTaskScheduler::GetInstance().SetSkipDepth(SkipDepth);
}

void ULlamaDialogueWidget::SetSkipPolicy(const int SkipPolicy)
{
	FTaskScheduler::GetInstance().SetSkipPolicy(SkipPolicy);
}

void ULlamaDialogueWidget::SetOpPolicy(const int OpPolicy)
{
	FTaskScheduler::GetInstance().SetOpPolicy(OpPolicy);
	FPerformanceDetector::GetInstance().SetOpPolicy(OpPolicy);
}


void ULlamaDialogueWidget::SetAPM(const int nAPM)
{
	APM = nAPM;
	FPerformanceDetector::GetInstance().SetTargetAPM(APM);
}


int ULlamaDialogueWidget::LastNumberOfCopies = 0;


void ULlamaDialogueWidget::AddObjects(int NumberOfCopies, UStaticMesh* StaticMesh)
{
	if (!StaticMesh) return;

	int TargetNumberOfCopies = NumberOfCopies;
	NumberOfCopies -= LastNumberOfCopies;
	LastNumberOfCopies = TargetNumberOfCopies;

	if (NumberOfCopies > 0)
	{
		// FVector CameraLocation = GetWorld()->GetFirstPlayerController()->PlayerCameraManager->GetCameraLocation();
		// FRotator CameraRotation = GetWorld()->GetFirstPlayerController()->PlayerCameraManager->GetCameraRotation();

		// 获取玩家控制的人物
		APlayerController* PlayerController = GetWorld()->GetFirstPlayerController();
		if (!PlayerController) return;

		APawn* ControlledPawn = PlayerController->GetPawn();
		if (!ControlledPawn) return;

		FVector PawnLocation = ControlledPawn->GetActorLocation();
		FRotator PawnRotation = ControlledPawn->GetActorRotation();
		
		for (int i = 0; i < NumberOfCopies; ++i)
		{
			// FVector SpawnLocation = CameraLocation + CameraRotation.Vector() * 200.0f + 100.0f;
			// FRotator SpawnRotation = CameraRotation;
			
			// 计算生成物体的位置
			FVector SpawnLocation = PawnLocation + PawnRotation.Vector() * 200.0f + FMath::VRand() * 50.0f;
			FRotator SpawnRotation = PawnRotation;

			AActor* TempActor = GetWorld()->SpawnActor<AActor>(AActor::StaticClass(), SpawnLocation, SpawnRotation);

			if (TempActor)
			{
				UStaticMeshComponent* MeshComponent = NewObject<UStaticMeshComponent>(TempActor);
				MeshComponent->SetStaticMesh(StaticMesh);
				//MeshComponent->SetWorldLocationAndRotation(SpawnLocation, SpawnRotation);
				MeshComponent->RegisterComponent();
				TempActor->AddInstanceComponent(MeshComponent);
				AddedMeshComponents.Add(MeshComponent);
				AddedActors.Add(TempActor);
			}
		}
	}
	else
	{
		NumberOfCopies = - NumberOfCopies;
		for (int32 i = 0; i < NumberOfCopies && AddedMeshComponents.Num() > 0; ++i)
		{
			UStaticMeshComponent* MeshComponent = AddedMeshComponents.Pop();
			if (MeshComponent)
			{
				AActor* OwnerActor = MeshComponent->GetOwner();
				MeshComponent->DestroyComponent();
				if (OwnerActor)
				{
					OwnerActor->Destroy(); // 销毁组件的所有者AActor
					AddedActors.Remove(OwnerActor); // 从记录中移除已销毁的AActor
				}
				GetWorld()->ForceGarbageCollection(true);
			}
		}
	}
}
















