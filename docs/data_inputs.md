# Data Inputs and Zarr Stores

## Zarr stores

Stores follow the layout in `data_processor::store_schema`, written by `data_processor::output` natively and by
`data_processor::zarr_writer` in the browser. Stores of another `avalanchers_format` are rejected:

```
<store>.zarr/
  <site>/            # group, one per site
    dem, x, y        # arrays: the terrain of the site
    <scenario>/      # group, one per scenario
      release_area   # array: the release areas of the scenario
      peak_flow_velocity, peak_flow_thickness  # (run, y, x)
      center_of_mass_x, center_of_mass_y, time # (run, timestep), indexed by step, NaN without enable_center_of_mass
      travel_length_2d, travel_length_3d, travel_vertical_drop, travel_angle  # (run): center-of-mass trajectory
      friction_coefficient, cfl, created_at, ...  # (run): results, provenance, every setting
```

The `run` dimension grows in blocks, so only the first `number_of_runs` entries (a scenario attribute) are runs;
`AvalanchersScenario.dataset` already cuts it to those. `x` and `y` are cell centers. Each save holds a lock in the file
`<store>.zarr.lock` next to the store, so several processes can write to one store.

See [model conventions](model_conventions.md#center-of-mass-travel-metrics) for how the travel metrics are computed.

### Grid orientation

Grids keep row 0 at the south edge, so `y` ascends with the rows, in the store as well as in arrays such as
`sim.peak_velocity`. Formats that list the north row first are flipped when they are read and written, see
`sim.write_asc`. Use `.sortby("y", ascending=False)` on an xarray dataset before exporting it as a north-up raster.

### Stores as inputs

In the settings, `dem_path` accepts `<store>.zarr`, `<store>.zarr/<site>` or
`<store>.zarr/<site>/<scenario>`; a site or scenario can be named in full, without its hash, or by its hash, and may be
left out if it is the only one. The release areas of the selected scenario are used too, unless `release_areas_path` is
set, which can point into a store as well, or `release_area_fraction` is set and no scenario was named. A site with
several scenarios is an error unless one is named or another release is set; a site with none gets calculated release
areas and a warning. Saving into the same store continues the scenario.

```json
{ "dem_path": "avalanchers.zarr/avaWog/avaWogreleaseTexture" }
```

### Browser decoding

Directory uploads use the browser's `webkitdirectory` support, so the whole store folder has to be selected
(dropping the folder onto the drop zone works as well).

Chunks are decoded by the WASM build. The blosc codec (zstd compressor, with the byte-shuffle or bitshuffle
filter) is implemented in pure Rust in `data_processor::blosc`, using [`ruzstd`](https://crates.io/crates/ruzstd)
rather than the C `zstd` bindings, because `zstd-sys` needs a clang cross-compiler to target
`wasm32-unknown-unknown`. Half precision (`float16`) arrays are widened to `f32` when read.

## Release areas

`release_areas_path` also accepts a polygon shapefile (`.shp`) in the coordinate system of the DEM, native and Python
only. All its features are rasterized onto the DEM grid: cells whose center lies inside a polygon release, holes stay
free. As in AvaFrame, the thickness of each feature comes from its `thickness` or `d0` attribute, either all features
or none must have one, and features must not overlap. Without these attributes the thickness is 1 m.

`release_thickness` (m, unset by default) replaces the thickness of every release cell, whatever the release source,
and logs a warning if that source has a thickness of its own.

`release_swe` (kg/m², the same as mm of water, unset by default) can be set instead of `release_thickness`; setting
both is an error. It must be positive, and `release_swe / density` must be above 0.01 m. It is the mass per horizontal
area, so a release cell holds `release_swe * cell_size²` of snow whatever its slope, while `release_thickness` is
measured normal to the slope.
