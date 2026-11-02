# =============================================================================
#  Tai ban do duong bo thuc te DIMACS 9th Challenge (USA-road-d) va giai nen .gz
#  Dung:  powershell -ExecutionPolicy Bypass -File scripts\download_dimacs.ps1 NY
#  Vung:  NY (264K dinh) | BAY (321K) | COL (435K) | FLA (1.07M) | NW (1.2M) | NE (1.5M) | CAL (1.9M)
# =============================================================================
param([string]$Region = "NY")
$base = "http://www.diag.uniroma1.it/challenge9/data/USA-road-d"
$dir = Join-Path $PSScriptRoot "..\data"
New-Item -ItemType Directory -Force -Path $dir | Out-Null

function Expand-Gz($src, $dst) {
    $in = [System.IO.File]::OpenRead($src)
    $gz = New-Object System.IO.Compression.GZipStream($in, [System.IO.Compression.CompressionMode]::Decompress)
    $out = [System.IO.File]::Create($dst)
    $gz.CopyTo($out)
    $out.Close(); $gz.Close(); $in.Close()
}

foreach ($ext in @("gr", "co")) {
    $name = "USA-road-d.$Region.$ext"
    $gzPath = Join-Path $dir "$name.gz"
    $outPath = Join-Path $dir $name
    if (Test-Path $outPath) { Write-Host "Exists: $outPath"; continue }
    Write-Host "Downloading $base/$name.gz ..."
    Invoke-WebRequest -Uri "$base/$name.gz" -OutFile $gzPath
    Expand-Gz $gzPath $outPath
    Remove-Item $gzPath
    Write-Host "OK: $outPath"
}
Write-Host ""
Write-Host "Run:  pathfinder.exe --mode bench --gr data\USA-road-d.$Region.gr --co data\USA-road-d.$Region.co --queries 5"
