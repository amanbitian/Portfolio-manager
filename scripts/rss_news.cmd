@echo off
REM Poll Indian financial-media + BSE-notice RSS feeds and append to the data lake.
REM Registered as a Windows Scheduled Task by scripts\install_rss_schedule.ps1.
REM Output goes to the pipeline's own _logs\news-<date>.log.

cd /d "F:\quants project\portfolio manager"

REM Full path so it works in Task Scheduler's minimal PATH.
set "PY=C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe"
if exist "%PY%" (
  "%PY%" -m data_pipeline.run news --source rss
) else (
  py -3 -m data_pipeline.run news --source rss
)
exit /b %ERRORLEVEL%
