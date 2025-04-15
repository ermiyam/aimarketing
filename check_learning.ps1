# Read the process ID from the file
$pidFile = "logs/learning_pid.txt"
if (Test-Path $pidFile) {
    $processId = Get-Content $pidFile
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    
    if ($process) {
        Write-Host "Mak's learning process is running (PID: $processId)"
        Write-Host "Process started at: $($process.StartTime)"
        Write-Host "CPU Usage: $($process.CPU) %"
        Write-Host "Memory Usage: $([math]::Round($process.WorkingSet64 / 1MB, 2)) MB"
        
        # Check log file
        $logFile = "logs/continuous_learn.log"
        if (Test-Path $logFile) {
            $lastLog = Get-Content $logFile -Tail 5
            Write-Host "`nLast 5 log entries:"
            $lastLog | ForEach-Object { Write-Host $_ }
        }
    } else {
        Write-Host "Process with PID $processId is not running"
        Remove-Item $pidFile
    }
} else {
    Write-Host "No learning process found. PID file does not exist."
} 