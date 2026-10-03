@echo off
echo Starting Face2Health Mockup Dashboard...

:: Ensure conda face2bmi environment is active if streamlit is not in current PATH
where streamlit >nul 2>&1
if %errorlevel% neq 0 (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" face2bmi 2>nul || call "%USERPROFILE%\anaconda3\Scripts\activate.bat" face2bmi 2>nul || call conda activate face2bmi 2>nul
)

streamlit run dashboard/app.py
pause

