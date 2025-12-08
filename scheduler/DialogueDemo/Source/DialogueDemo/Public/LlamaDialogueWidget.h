// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "SchedulingSystem/TaskScheduler/Public/TaskScheduler.h"
#include "LlamaInferenceTask.h"
#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "LlamaDialogueWidget.generated.h"

#define TARGET_INFERENCE_SPEED 45

/**
 * 
 */
UCLASS()
class DIALOGUEDEMO_API ULlamaDialogueWidget : public UUserWidget
{
	GENERATED_BODY()

public:
	
	std::string Prompt;
	
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Llama Dialogue")
	FString Content;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Llama Dialogue")
	int32 Order;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Llama Dialogue")
	int32 APM; //time per action
	
	ULlamaDialogueWidget(const FObjectInitializer& ObjectInitializer);
	
	UFUNCTION(BlueprintCallable, Category="Llama Dialogue")
	bool GetNextPrompt();

	UFUNCTION(BlueprintCallable, Category="Llama Dialogue")
	void LlamaInference();

	UFUNCTION(BlueprintCallable, Category="Game Performance")
	float GetAverageFPS();

	UFUNCTION(BlueprintCallable, Category="Game Performance")
	float GetInferenceTime();

	UFUNCTION(BlueprintCallable, Category="Game Performance")
	float GetTotalInferenceTime();

	UFUNCTION(BlueprintCallable, Category="ObjectManagement")
	void AddObjects(int NumberOfCopies, UStaticMesh* StaticMesh);

	UFUNCTION(BlueprintCallable, Category="Llama Dialogue")
	void SetSkipDepth(const int SkipDepth);

	UFUNCTION(BlueprintCallable, Category="Llama Dialogue")
	void SetSkipPolicy(const int SkipPolicy);

	UFUNCTION(BlueprintCallable, Category="Llama Dialogue")
	void SetOpPolicy(const int OpPolicy);

	UFUNCTION(BlueprintCallable, Category="Llama Dialogue")
	void SetAPM(const int nAPM);
	
private:
	FString prompts[5];
	
	bool AutoSchedule;

	TArray<UStaticMeshComponent*> AddedMeshComponents;
	TArray<AActor*> AddedActors;
	static int LastNumberOfCopies;
};


