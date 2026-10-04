@echo off
setlocal enabledelayedexpansion

IF NOT EXIST tests_cpp.generated goto Create
IF NOT EXIST tests_unit.generated goto Create
IF NOT EXIST tests_gtest.generated goto Create

Set "orig=tests.list"
Set "gen=tests_cpp.generated"

Set "orig_ModDateTime="
Set "gen_ModDateTime="

:: Retrieve the modification date and time of the original file
for /f "delims=" %%i in ('wmic datafile where name^="%cd:\=\\%\\%orig%" get lastmodified /format:list ^| find "="') do (
    for /f "tokens=2 delims==" %%j in ("%%i") do set "orig_ModDateTime=%%j"
)

:: Retrieve the modification date and time of the generated file
for /f "delims=" %%i in ('wmic datafile where name^="%cd:\=\\%\\%gen%" get lastmodified /format:list ^| find "="') do (
    for /f "tokens=2 delims==" %%j in ("%%i") do set "gen_ModDateTime=%%j"
)

:: Display results for verification
echo Original File: %orig_ModDateTime%
echo Generated File: %gen_ModDateTime%

:: Compare timestamps
if "%orig_ModDateTime%" LSS "%gen_ModDateTime%" goto Message

:Create
echo Generating Files
> tests_cpp.generated echo.
> tests_unit.generated echo.
> tests_gtest.generated echo.
start ../../bin/Release/es_test_app.exe "-b" tests.list tests_cpp.generated tests_unit.generated tests_gtest.generated
goto End

:Message
echo The list has not been modified.

:End
