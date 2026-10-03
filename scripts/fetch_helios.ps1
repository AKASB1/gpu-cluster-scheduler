# Downloads the Helios GPU-cluster traces (S-Lab-System-Group/HeliosData,
# CC-BY-4.0) into the ignored benchmarks\outputs\helios\ and extracts the two
# Venus files the replay scenario (S13) reads. Run from the repository root:
#     powershell -ExecutionPolicy Bypass -File scripts\fetch_helios.ps1
# Nothing from the traces is committed; the replay results cite the source.
$ErrorActionPreference = 'Stop'
$dest = 'benchmarks\outputs\helios'
$url = 'https://raw.githubusercontent.com/S-Lab-System-Group/HeliosData/master/data.zip'
$want = '3d22a5f6c0ae669e2fcbfe4200fa9c48664507bc397c677bad8f085222c032ac'
New-Item -ItemType Directory -Force $dest | Out-Null
$zip = Join-Path $dest 'data.zip'
if (-not (Test-Path $zip)) {
    Invoke-WebRequest -Uri $url -OutFile "$zip.part" -UseBasicParsing -TimeoutSec 900
    Move-Item "$zip.part" $zip
}
$h = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
if ($h -ne $want) { throw "data.zip has SHA-256 $h, expected $want" }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $zip))
try {
    foreach ($name in @('data/Venus/cluster_log.csv', 'data/Venus/cluster_gpu_number.csv')) {
        $entry = $archive.GetEntry($name)
        $target = Join-Path $dest ($name -replace '/', '\')
        New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
    }
} finally {
    $archive.Dispose()
}
Write-Output "extracted the Venus files into $dest\data\Venus (CC-BY-4.0, S-Lab-System-Group/HeliosData)"
