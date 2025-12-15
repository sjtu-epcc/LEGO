using UnrealBuildTool;

public class PerformanceDetector: ModuleRules
{
	public PerformanceDetector(ReadOnlyTargetRules Target) : base(Target)
	{
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "Engine", "TaskDetector"});
		
		PrivateIncludePaths.Add("Runtime\\SchedulingSystem\\PerformanceDetector\\Private");
		
		bEnableUndefinedIdentifierWarnings = false;
	}
}