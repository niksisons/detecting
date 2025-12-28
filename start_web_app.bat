@echo off
chcp 65001 >nul
echo.
echo ========================================
echo 🎯 Запуск веб-интерфейса мониторинга
echo ========================================
echo.

cd /d "%~dp0"

REM Активация виртуального окружения
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo ❌ Виртуальное окружение не найдено!
    echo Создайте его командой: python -m venv .venv
    pause
    exit /b 1
)

REM Проверка установки streamlit
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo 📦 Установка необходимых пакетов...
    pip install streamlit plotly openpyxl
)

echo.
echo 🚀 Запуск Streamlit...
echo.
echo 📍 Откройте в браузере: http://localhost:8501
echo.
echo ⏹️ Для остановки нажмите Ctrl+C
echo.

streamlit run web_app.py

pause
