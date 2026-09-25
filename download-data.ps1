$ErrorActionPreference = 'Stop'
$rawFolder = Join-Path $PSScriptRoot 'data\raw'
New-Item -ItemType Directory -Path $rawFolder -Force | Out-Null
$revision = '3c8bef32a2b6cb1f70d686783edeecaec6287cbf'
$baseUrl = "https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews/resolve/$revision"
Invoke-WebRequest -Uri "$baseUrl/README.md" -OutFile (Join-Path $rawFolder 'SOURCE_README.md')
Invoke-WebRequest -Uri "$baseUrl/data/train-00000-of-00001.parquet" -OutFile (Join-Path $rawFolder 'reviews.parquet')
Write-Output 'Veri indirildi. Ücretli model çağrısı yapılmadı.'
