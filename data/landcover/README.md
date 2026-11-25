# Landcover Data

1. Download from CLC backbone data from [https://land.copernicus.eu/en/products/clc-backbone](https://land.copernicus.eu/en/products/clc-backbone). Available for EU and some other countries.
2. Create tiles with `create_landcover_tiles.ps1`

Categories:

| Index | Name                              | R   | G   | B   |
| ----- | --------------------------------- | --- | --- | --- |
| 1     | Sealed                            | 255 | 0   | 0   |
| 2     | Woody needle leaved trees         | 34  | 139 | 34  |
| 3     | Woody broadleaved deciduous trees | 128 | 255 | 0   |
| 4     | Woody broadleaved evergreen trees | 0   | 255 | 8   |
| 5     | Low-growing woody plants          | 128 | 64  | 0   |
| 6     | Permanent herbaceous              | 204 | 242 | 77  |
| 7     | Periodically herbaceous           | 255 | 255 | 128 |
| 8     | Lichens and mosses                | 255 | 128 | 255 |
| 9     | Non and sparsely vegetated        | 191 | 191 | 191 |
| 10    | Water                             | 0   | 128 | 255 |
| 11    | Snow and ice                      | 0   | 255 | 255 |
| 253   | Coastal seawater buffer           | 191 | 223 | 255 |
| 254   | Outside area                      | 230 | 230 | 230 |
| 255   | No data                           | 0   | 0   | 0   |

## Forest cover for the forest model

`forest_cover_from_clcplus.py` turns the CLCplus download zip into a canopy cover raster for the
`forest_cover_path` setting. The cover of a cell is the area fraction of tree pixels (classes 2-4),
warped onto the DEM grid with average resampling and smoothed with a Gaussian (`--smoothing`, default 10 m):

```
python data/landcover/forest_cover_from_clcplus.py --reference dem.tif --output forest_cover.tif
python data/landcover/forest_cover_from_clcplus.py --epsg 2056 --bounds 2780000 1180000 2786000 1186000 --cell-size 5 --output forest_cover.tif
```

It needs `rasterio` (and `scipy` for the smoothing) and reads the 100 km tiles it needs directly from
`CLCPLUS_RAS_S2023_R10m_03035.zip`.
