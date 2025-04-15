# Start the continuous learning process
$process = Start-Process python -ArgumentList "src/continuous_learn.py" -WindowStyle Hidden -PassThru

# Save the process ID to a file for later reference
$process.Id | Out-File -FilePath "logs/learning_pid.txt"

Write-Host "Mak's continuous learning process started with PID: $($process.Id)"
Write-Host "Check logs/continuous_learn.log for progress"
Write-Host "To stop the process, run: Stop-Process -Id $($process.Id)" 