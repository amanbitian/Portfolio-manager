# Register (or refresh) a Windows Scheduled Task that polls the RSS news feeds
# every 2 hours while the machine is on. Run this once, from an ordinary
# (non-admin) PowerShell:
#
#     powershell -ExecutionPolicy Bypass -File scripts\install_rss_schedule.ps1
#
# Manage afterwards:
#     schtasks /query /tn "QuantsRssNews"        # show
#     schtasks /run   /tn "QuantsRssNews"        # run now
#     schtasks /delete /tn "QuantsRssNews" /f    # remove
#     (or use Task Scheduler GUI -> Task Scheduler Library)

$TaskName = "QuantsRssNews"
$Script   = "F:\quants project\portfolio manager\scripts\rss_news.cmd"
$IntervalHours = 2

if (-not (Test-Path $Script)) { throw "missing $Script" }

$action  = New-ScheduledTaskAction -Execute $Script
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddMinutes(5) `
             -RepetitionInterval (New-TimeSpan -Hours $IntervalHours)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
             -DontStopOnIdleEnd -MultipleInstances IgnoreNew `
             -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Poll Indian financial RSS feeds for the quant data lake" `
    -Force | Out-Null

Write-Host "Registered '$TaskName' - runs every $IntervalHours h while the PC is on."
Write-Host "Missed runs fire on next wake (StartWhenAvailable). Test now:  schtasks /run /tn $TaskName"
