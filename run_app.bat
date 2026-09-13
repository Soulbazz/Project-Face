@echo off
title Face-to-BMI & NCDs Risk Screening System
echo ========================================================
echo   Launching AI Health Screening Application...
echo ========================================================

:: เรียกใช้งาน Conda Command
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" face2bmi 2>nul || call "%USERPROFILE%\anaconda3\Scripts\activate.bat" face2bmi 2>nul || call conda activate face2bmi

:: ตรวจสอบการเปิดใช้งาน Environment
if %errorlevel% neq 0 (
    echo [ERROR] Cannot activate conda environment 'face2bmi'.
    echo Please make sure Conda is installed and environment is created.
    pause
    exit /b
)

:: ย้ายเข้าโฟลเดอร์ scripts แล้วรัน Streamlit
cd scripts
echo Starting Streamlit server...
streamlit run app.py

pause