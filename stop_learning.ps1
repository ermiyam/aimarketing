# Read the process ID from the file
$pidFile = "logs/learning_pid.txt"
if (Test-Path $pidFile) {
    $processId = Get-Content $pidFile
    try {
        Stop-Process -Id $processId -Force
        Write-Host "Successfully stopped Mak's learning process (PID: $processId)"
        Remove-Item $pidFile
    } catch {
        Write-Host "Error stopping process: $_"
    }
} else {
    Write-Host "No learning process found. PID file does not exist."
} 