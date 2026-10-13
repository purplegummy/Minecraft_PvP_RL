# Compiles the plugin against Paper's API, its libraries and the server internals (fake players need them),
# then copies the jar into server/plugins.
$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$server = Resolve-Path (Join-Path $root '..\server')
$build = Join-Path $root 'build'
$classes = Join-Path $build 'classes'

if (Test-Path $build) { Remove-Item $build -Recurse -Force }
New-Item -ItemType Directory -Force $classes | Out-Null

function Fwd($p) { ($p -replace '\\', '/') }

$jars = @(Get-ChildItem (Join-Path $server 'versions') -Recurse -Filter *.jar) +
        @(Get-ChildItem (Join-Path $server 'libraries') -Recurse -Filter *.jar)
$classpath = ($jars | ForEach-Object { Fwd $_.FullName }) -join ';'
$sources = Get-ChildItem (Join-Path $root 'src') -Recurse -Filter *.java | ForEach-Object { '"' + (Fwd $_.FullName) + '"' }

# javac reads its arguments from a file so the long classpath doesn't hit the Windows command-line limit
$argfile = Join-Path $build 'javac.args'
@('-d', ('"' + (Fwd $classes) + '"'), '--release', '25', '-cp', ('"' + $classpath + '"')) + $sources |
    Set-Content $argfile -Encoding ascii

& javac "@$argfile"
if ($LASTEXITCODE -ne 0) { throw 'javac failed' }

Copy-Item (Join-Path $root 'plugin.yml') $classes
# tells Paper this jar already uses Mojang names for server internals, so it must not remap it
$manifest = Join-Path $build 'MANIFEST.MF'
"paperweight-mappings-namespace: mojang`n" | Set-Content $manifest -Encoding ascii -NoNewline
$jarPath = Join-Path $build 'PvpEnv.jar'
& jar --create --file $jarPath --manifest $manifest -C $classes .
if ($LASTEXITCODE -ne 0) { throw 'jar failed' }

Copy-Item $jarPath (Join-Path $server 'plugins\PvpEnv.jar') -Force
Write-Host 'Built plugin and copied it to server/plugins/PvpEnv.jar'
