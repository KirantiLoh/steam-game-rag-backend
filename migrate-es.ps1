param()

$esContainer = "elasticsearch"
$volumeName = "steamrec_es_data"

Write-Output "=== Migrate existing ES data to Docker volume ==="

# Check if source container exists
$running = docker ps --filter "name=$esContainer" --format "{{.Names}}" 2>$null
if (-not $running) {
    Write-Output "WARNING: Container '$esContainer' is not running."
    Write-Output "Fallback: compose up will auto-index (~2 min)."
    exit 0
}

Write-Output "1. Stopping current $esContainer..."
docker stop $esContainer 2>$null

Write-Output "2. Creating volume $volumeName..."
docker volume create $volumeName 2>$null

Write-Output "3. Copying ES data to volume..."
# Use a temporary alpine container to copy data
docker run --rm `
    --volumes-from $esContainer `
    -v "${volumeName}:/target" `
    alpine:latest ash -c "cp -r /usr/share/elasticsearch/data/* /target/" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Output "4. Migration complete! Volume '$volumeName' has the data."
    Write-Output "   Remove old container: docker rm $esContainer"
    Write-Output "   Then: docker compose up"
} else {
    Write-Output "ERROR: Migration failed. Fallback to auto-index on compose up."
}
