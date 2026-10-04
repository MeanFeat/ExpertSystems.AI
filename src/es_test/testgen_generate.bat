@echo off
setlocal enabledelayedexpansion

set "regen="

:: Any generated file missing -> regenerate
for %%G in (tests_cpp.generated tests_unit.generated tests_gtest.generated) do (
    if not exist "%%G" set "regen=1"
)

:: Any generated file older than tests.list -> regenerate
if not defined regen (
    for %%G in (tests_cpp.generated tests_unit.generated tests_gtest.generated) do (
        set "newest="
        for /f "delims=" %%N in ('dir /b /o:d "tests.list" "%%G"') do set "newest=%%N"
        if /i "!newest!"=="tests.list" set "regen=1"
    )
)

if not defined regen (
    echo The list has not been modified.
    goto :eof
)

echo Generating Files
> tests_cpp.generated echo.
> tests_unit.generated echo.
> tests_gtest.generated echo.
start ../../bin/Release/es_test_app.exe "-b" tests.list tests_cpp.generated tests_unit.generated tests_gtest.generated