using UnrealBuildTool;

public class NewLlamaModel: ModuleRules
{
	public NewLlamaModel(ReadOnlyTargetRules Target) : base(Target)
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
		
		PrivateIncludePaths.Add("Runtime\\NewLlamaModel\\Private");

		PrivateIncludePaths.Add("Runtime\\NewLlamaModel\\Private\\llamacpp");

		PrivateIncludePaths.Add("Runtime\\NewLlamaModel\\Private\\llamacpp\\common");
		
		PrivateIncludePaths.Add("E:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\include");
		
		PublicAdditionalLibraries.Add("Runtime\\NewLlamaModel\\Private\\Libs\\cudart.lib");
		
		PublicAdditionalLibraries.Add("Runtime\\NewLlamaModel\\Private\\Libs\\cublas.lib");
		
		PublicAdditionalLibraries.Add("Runtime\\NewLlamaModel\\Private\\Libs\\cuda.lib");
		
		PublicAdditionalLibraries.Add("Runtime\\NewLlamaModel\\Private\\Libs\\ggml.lib");
		
		bEnableUndefinedIdentifierWarnings = false;
	}
}