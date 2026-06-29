$ErrorActionPreference = 'SilentlyContinue'
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object {
        $_.CommandLine -match 'uvicorn.*worldcup\.api\.main|streamlit run.*dashboard\\app\.py|scripts\.serve'
    } |
    ForEach-Object {
        Write-Host "       结束 python PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force
    }
