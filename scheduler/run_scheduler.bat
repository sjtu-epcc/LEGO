@echo off

REM ------------------------------------------------------------
REM Usage:
REM   run_ue_llm.bat <game_trace> <llm_model>
REM
REM Arguments:
REM   game_trace : game rendering trace to replay
REM   llm_model  : large language model used for competition
REM ------------------------------------------------------------

set GAME_TRACE=%1
set LLM_MODEL=%2

REM BootstrapPackagedGame.exe is the official UnrealEngine
REM runtime entrypoint for packaged (non-editor) execution.

UnrealEngine\Engine\Binaries\Win64\BootstrapPackagedGame.exe ^
  -ReplayTrace=%GAME_TRACE% ^
  -LLMModel=%LLM_MODEL% ^
  -EnableLLMSync=1