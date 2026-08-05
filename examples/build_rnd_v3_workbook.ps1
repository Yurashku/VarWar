param(
    [string]$Payload = "results/rnd_v3/smoke/excel_payload.json",
    [string]$Output = "outputs/rnd-v3/VarWar_RnD_v3.xlsx",
    [string]$PreviewDir = "outputs/rnd-v3/previews"
)

$started = Get-Date
& node "$PSScriptRoot/build_rnd_v3_workbook.mjs" $Payload $Output $PreviewDir
$nodeExit = $LASTEXITCODE
$workbook = Get-Item -LiteralPath $Output -ErrorAction SilentlyContinue
$previews = @(Get-ChildItem -LiteralPath $PreviewDir -Filter "*.png" -ErrorAction SilentlyContinue)

# artifact-tool may report a native-worker teardown code on Windows after all
# awaited files have been flushed. Treat the build as successful only when this
# invocation produced a non-empty workbook and all eight requested previews.
if (
    $workbook -and
    $workbook.Length -gt 0 -and
    $workbook.LastWriteTime -ge $started.AddSeconds(-2) -and
    $previews.Count -eq 8
) {
    exit 0
}

exit $nodeExit
