using UnrealBuildTool;

public class LlamaModel: ModuleRules
{
	public LlamaModel(ReadOnlyTargetRules Target) : base(Target)
	{
	
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"Core", 
				"Engine", 
				"DurationPredictor", 
				"PerformanceDetector", 
				"TaskDetector", 
				"TaskScheduler",
			});

		PublicDefinitions.Add("GGML_USE_CUBLAS=1");
		
		PublicDefinitions.Add("GGML_CUDA_F16=1");
		
		PrivateIncludePaths.Add("Runtime\\LlamaModel\\Private");

		PrivateIncludePaths.Add("Runtime\\LlamaModel\\Private\\llamacpp");

		PrivateIncludePaths.Add("Runtime\\LlamaModel\\Private\\llamacpp\\common");
		
		PrivateIncludePaths.Add("E:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\include");
		
		PublicAdditionalLibraries.Add("Runtime\\LlamaModel\\Private\\Libs\\cudart.lib");
		
		PublicAdditionalLibraries.Add("Runtime\\LlamaModel\\Private\\Libs\\cublas.lib");
		
		PublicAdditionalLibraries.Add("Runtime\\LlamaModel\\Private\\Libs\\cuda.lib");
		
		PublicAdditionalLibraries.Add("Runtime\\LlamaModel\\Private\\Libs\\ggml.lib");
		
		bEnableUndefinedIdentifierWarnings = false;
	}
}