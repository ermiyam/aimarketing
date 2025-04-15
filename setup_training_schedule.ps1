# Create a scheduled task to run the scraper and training
$action = New-ScheduledTaskAction -Execute "python" -Argument "src/scraper.py" -WorkingDirectory (Get-Location)
$trigger = New-ScheduledTaskTrigger -Daily -At 12AM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 3
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

# Register the task
Register-ScheduledTask -TaskName "MakAutoTraining" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Automated content scraping and training for Mak AI"

Write-Host "✅ Scheduled task created successfully!"
Write-Host "Task will run daily at midnight and retry up to 3 times if it fails." 