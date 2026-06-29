param(
    [string]$Url = "http://127.0.0.1:8000/health",
    [int]$MaxWaitSec = 120
)

$deadline = (Get-Date).AddSeconds($MaxWaitSec)
$attempt = 0
while ((Get-Date) -lt $deadline) {
    $attempt++
    try {
        $r = Invoke-RestMethod -Uri $Url -TimeoutSec 5
        if ($r.status -eq "ok") {
            Write-Host "       API ready (attempt $attempt)"
            exit 0
        }
    } catch {
        if ($attempt -eq 1 -or ($attempt % 10) -eq 0) {
            Write-Host "       still waiting... (${attempt}s)"
        }
    }
    Start-Sleep -Seconds 1
}
exit 1
