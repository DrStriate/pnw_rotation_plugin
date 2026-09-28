@echo off
echo Cleaning __pycache__ directories...

:: Loop through all subdirectories recursively looking for __pycache__
FOR /d /r . %%d in (__pycache__) do (
    if exist "%%d" (
        echo Deleting: %%d
        rd /s /q "%%d"
    )
)

echo.
echo Cleanup complete!
