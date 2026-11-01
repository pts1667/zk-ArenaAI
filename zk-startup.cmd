@echo off
rem Runs zk-startup.ps1 from cmd.exe, e.g.: zk-startup.cmd ControllerAI CircuitAI:Hard -Headless -Client "python my_bot.py {team}"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0zk-startup.ps1" %*
