@echo off
REM ============================================================
REM Configure UnrealEngine scheduler for LLM-rendering sync
REM This script prepares engine and project sources on Windows.
REM ============================================================

REM -------- User-configurable paths (relative or absolute) ----
set PROJECT_PATH=YourUnrealProjectPath
set UE_PATH=UEPath

REM -------- Clone UnrealEngine source (if not exists) ---------
REM NOTE: This step is optional for AE demonstration purposes.
git clone https://github.com/folgerwang/UnrealEngine

REM -------- Copy project source files -------------------------
REM Copy DialogueDemo project sources into the target project

xcopy /E /I /Y ^
  DialogueDemo\Source\DialogueDemo\* ^
  %PROJECT_PATH%\DialogueDemo\Source\DialogueDemo\

REM -------- Patch UnrealEngine runtime sources ----------------
REM Copy modified scheduler/runtime components into UE source tree

xcopy /E /I /Y ^
  UnrealEngine\Engine\Source\Runtime\* ^
  %UE_PATH%\UnrealEngine\Engine\Source\Runtime\
