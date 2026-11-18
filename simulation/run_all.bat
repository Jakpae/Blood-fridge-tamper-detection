@echo off
chcp 65001 > nul
cd /d %~dp0
call .venv\Scripts\activate.bat
python step1_generate.py || goto :fail
python step2_train.py || goto :fail
python step3_evaluate.py || goto :fail
python step4_plots.py || goto :fail
echo.
echo 모든 단계 완료. results 폴더를 확인하세요.
pause
exit /b 0
:fail
echo.
echo 오류로 중단되었습니다. 위 메시지를 확인하세요.
pause
exit /b 1
