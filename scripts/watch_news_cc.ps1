# Watch the Common Crawl news fetch. Refreshes every 20s.
#   powershell -ExecutionPolicy Bypass -File scripts\watch_news_cc.ps1
# Ctrl+C to stop watching (does not affect the fetch).

$root = "F:\quants project\stock Data\news\commoncrawl"
$log  = "F:\quants project\stock Data\_logs\news_cc_fetch.out"
$py   = "C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe"

$prevCount = $null
$prevTime  = $null

while ($true) {
    Clear-Host
    Write-Host ("Common Crawl news fetch  -  {0}" -f (Get-Date -Format "HH:mm:ss")) -ForegroundColor Cyan
    Write-Host ("-" * 52)

    # process alive?
    $proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -like '*news-cc*' }
    if ($proc) {
        $age = [int]((Get-Date) - $proc.CreationDate).TotalMinutes
        Write-Host ("process   : RUNNING  (PID {0}, {1} min)" -f $proc.ProcessId, $age) -ForegroundColor Green
    } else {
        Write-Host "process   : NOT RUNNING" -ForegroundColor Yellow
    }

    # saved article count (fast: count parquet row-groups via python)
    $count = & $py -c @"
import glob, pandas as pd
fs = glob.glob(r'$root/year=*/**/*.parquet', recursive=True)
n = 0
for f in fs:
    try: n += len(pd.read_parquet(f, columns=['url']))
    except Exception: pass
print(n)
"@ 2>$null
    $count = [int]$count

    $now = Get-Date
    if ($prevCount -ne $null) {
        $dt = ($now - $prevTime).TotalSeconds
        $rate = if ($dt -gt 0) { [math]::Round(($count - $prevCount) / $dt, 1) } else { 0 }
        Write-Host ("saved     : {0:N0} articles   (+{1}/s since last refresh)" -f $count, $rate)
    } else {
        Write-Host ("saved     : {0:N0} articles" -f $count)
    }
    $prevCount = $count
    $prevTime  = $now

    # progress-bar line from the log (has the tqdm rate + ETA)
    $bar = (Get-Content $log -Tail 1 -ErrorAction SilentlyContinue) -replace "`r", "`n"
    $bar = ($bar -split "`n" | Where-Object { $_ -match 'art/s|articles' } | Select-Object -Last 1)
    if ($bar) { Write-Host ("tqdm      : {0}" -f $bar.Trim()) -ForegroundColor DarkGray }

    # last real log line
    $lastLog = Get-Content $log -Tail 30 -ErrorAction SilentlyContinue |
               Where-Object { $_ -match 'INFO|WARNING' } | Select-Object -Last 1
    if ($lastLog) { Write-Host ("log       : {0}" -f $lastLog.Trim()) -ForegroundColor DarkGray }

    Start-Sleep -Seconds 20
}
