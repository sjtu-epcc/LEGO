// Copyright Epic Games, Inc. All Rights Reserved.
// 
// Engine module class

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class IRendererModule;

/** Implements the engine module. */
class FNewLlamaModule : public FDefaultModuleImpl
{
public:

	// IModuleInterface
	virtual void StartupModule() {}
	virtual void ShutdownModule() {}
};