//! Shapefile → `GeoPolygon`; `read_shapefile_nth_polygon` is what the harness calls per case.
use crate::rasterizer::{GeoPolygon, Point2D};
use shapefile::{PolygonRing, Shape, ShapeReader};
use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum ShapeError {
    #[error("Failed to open shapefile: {0}")]
    Open(#[from] shapefile::Error),
    #[error("No polygons found in shapefile")]
    NoPolygonsFound,
    #[error("Polygon index {index} out of bounds")]
    IndexOutOfBounds { index: u32 },
}

/// Reads all polygons from a shapefile and returns them as a single GeoPolygon containing all rings.
pub fn read_shapefile<P: AsRef<Path>>(path: P) -> Result<GeoPolygon, ShapeError> {
    let mut reader = ShapeReader::from_path(path)?;

    let mut all_rings = Vec::new();

    for result in reader.iter_shapes() {
        let shape = result?;
        match shape {
            Shape::Polygon(polygon) => {
                for ring in polygon.rings() {
                    let mut points = Vec::new();
                    let ring_points = match ring {
                        PolygonRing::Outer(pts) => pts,
                        PolygonRing::Inner(pts) => pts,
                    };
                    for p in ring_points {
                        points.push(Point2D { x: p.x, y: p.y });
                    }
                    all_rings.push(points);
                }
            }
            Shape::PolygonZ(polygon) => {
                for ring in polygon.rings() {
                    let mut points = Vec::new();
                    let ring_points = match ring {
                        PolygonRing::Outer(pts) => pts,
                        PolygonRing::Inner(pts) => pts,
                    };
                    for p in ring_points {
                        points.push(Point2D { x: p.x, y: p.y });
                    }
                    all_rings.push(points);
                }
            }
            Shape::PolygonM(polygon) => {
                for ring in polygon.rings() {
                    let mut points = Vec::new();
                    let ring_points = match ring {
                        PolygonRing::Outer(pts) => pts,
                        PolygonRing::Inner(pts) => pts,
                    };
                    for p in ring_points {
                        points.push(Point2D { x: p.x, y: p.y });
                    }
                    all_rings.push(points);
                }
            }
            _ => {}
        }
    }

    if all_rings.is_empty() {
        return Err(ShapeError::NoPolygonsFound);
    }

    Ok(GeoPolygon { rings: all_rings })
}

/// Reads all polygon features from a shapefile and returns a list of GeoPolygons.
pub fn read_shapefile_polygons<P: AsRef<Path>>(path: P) -> Result<Vec<GeoPolygon>, ShapeError> {
    let mut reader = ShapeReader::from_path(path)?;

    let mut geo_polygons = Vec::new();

    for result in reader.iter_shapes() {
        let shape = result?;
        let mut rings = Vec::new();
        match shape {
            Shape::Polygon(polygon) => {
                for ring in polygon.rings() {
                    let mut points = Vec::new();
                    let ring_points = match ring {
                        PolygonRing::Outer(pts) => pts,
                        PolygonRing::Inner(pts) => pts,
                    };
                    for p in ring_points {
                        points.push(Point2D { x: p.x, y: p.y });
                    }
                    rings.push(points);
                }
            }
            Shape::PolygonZ(polygon) => {
                for ring in polygon.rings() {
                    let mut points = Vec::new();
                    let ring_points = match ring {
                        PolygonRing::Outer(pts) => pts,
                        PolygonRing::Inner(pts) => pts,
                    };
                    for p in ring_points {
                        points.push(Point2D { x: p.x, y: p.y });
                    }
                    rings.push(points);
                }
            }
            Shape::PolygonM(polygon) => {
                for ring in polygon.rings() {
                    let mut points = Vec::new();
                    let ring_points = match ring {
                        PolygonRing::Outer(pts) => pts,
                        PolygonRing::Inner(pts) => pts,
                    };
                    for p in ring_points {
                        points.push(Point2D { x: p.x, y: p.y });
                    }
                    rings.push(points);
                }
            }
            _ => continue,
        }

        if !rings.is_empty() {
            geo_polygons.push(GeoPolygon { rings });
        }
    }

    Ok(geo_polygons)
}

/// Reads the nth polygon feature from a shapefile (0-indexed) and returns it as a GeoPolygon.
pub fn read_shapefile_nth_polygon<P: AsRef<Path>>(
    path: P,
    index: u32,
) -> Result<GeoPolygon, ShapeError> {
    let mut reader = ShapeReader::from_path(path)?;

    let mut current_idx = 0;
    for result in reader.iter_shapes() {
        let shape = result?;
        let mut rings = Vec::new();
        match shape {
            Shape::Polygon(polygon) => {
                for ring in polygon.rings() {
                    let mut points = Vec::new();
                    let ring_points = match ring {
                        PolygonRing::Outer(pts) => pts,
                        PolygonRing::Inner(pts) => pts,
                    };
                    for p in ring_points {
                        points.push(Point2D { x: p.x, y: p.y });
                    }
                    rings.push(points);
                }
            }
            Shape::PolygonZ(polygon) => {
                for ring in polygon.rings() {
                    let mut points = Vec::new();
                    let ring_points = match ring {
                        PolygonRing::Outer(pts) => pts,
                        PolygonRing::Inner(pts) => pts,
                    };
                    for p in ring_points {
                        points.push(Point2D { x: p.x, y: p.y });
                    }
                    rings.push(points);
                }
            }
            Shape::PolygonM(polygon) => {
                for ring in polygon.rings() {
                    let mut points = Vec::new();
                    let ring_points = match ring {
                        PolygonRing::Outer(pts) => pts,
                        PolygonRing::Inner(pts) => pts,
                    };
                    for p in ring_points {
                        points.push(Point2D { x: p.x, y: p.y });
                    }
                    rings.push(points);
                }
            }
            _ => continue,
        }

        if !rings.is_empty() {
            if current_idx == index {
                return Ok(GeoPolygon { rings });
            }
            current_idx += 1;
        }
    }

    Err(ShapeError::IndexOutOfBounds { index })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rasterizer::RasterGrid;
    use shapefile::dbase::TableWriterBuilder;
    use shapefile::{Point, Polygon, PolygonRing, Writer};
    use tempfile::tempdir;

    #[test]
    fn test_read_shapefile() {
        let dir = tempdir().expect("Failed to create temp dir");
        let shp_path = dir.path().join("test_polygon.shp");

        // Create a writer
        let table_builder = TableWriterBuilder::new();
        let mut writer =
            Writer::from_path(&shp_path, table_builder).expect("Failed to create shapefile writer");

        // Create a polygon
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
            Point::new(0.0, 0.0),
        ];
        let polygon = Polygon::with_rings(vec![PolygonRing::Outer(points)]);
        let record = shapefile::dbase::Record::default();

        writer
            .write_shape_and_record(&polygon, &record)
            .expect("Failed to write shape");
        drop(writer); // Flush/close files

        // Read it back with our reader
        let geo_poly = read_shapefile(&shp_path).expect("Failed to read shapefile");

        assert_eq!(geo_poly.rings.len(), 1);
        assert_eq!(geo_poly.rings[0].len(), 5);
        assert_eq!(geo_poly.rings[0][0].x, 0.0);
        assert_eq!(geo_poly.rings[0][0].y, 0.0);
        assert_eq!(geo_poly.rings[0][2].x, 10.0);
        assert_eq!(geo_poly.rings[0][2].y, 10.0);

        // Test reading multiple polygons
        let geo_polys =
            read_shapefile_polygons(&shp_path).expect("Failed to read shapefile polygons");
        assert_eq!(geo_polys.len(), 1);
        assert_eq!(geo_polys[0].rings.len(), 1);

        // Test reading nth polygon
        let geo_poly_0 =
            read_shapefile_nth_polygon(&shp_path, 0).expect("Failed to read 0th polygon");
        assert_eq!(geo_poly_0.rings.len(), 1);

        let geo_poly_err = read_shapefile_nth_polygon(&shp_path, 1);
        assert!(matches!(
            geo_poly_err,
            Err(ShapeError::IndexOutOfBounds { index: 1 })
        ));
    }
    #[test]
    #[ignore]
    fn test_read_shapefile_nth_polygon() {
        // let path = "../../data/avalanche_outlines_switzerland_2018/outlines2018.shp";
        let path = "../../data/single_avalanche_outlines_switzerland_2018/polygon_0.shp";
        let geo_poly = read_shapefile_nth_polygon(path, 0).expect("Failed to read 0th polygon");

        assert_eq!(geo_poly.rings.len(), 1);
        assert_eq!(geo_poly.rings[0].len(), 317);
        assert_eq!(geo_poly.rings[0][0].x, 2574616.6312000006);
        assert_eq!(geo_poly.rings[0][0].y, 1081003.3420000002);
        assert_eq!(geo_poly.rings[0][316].x, 2574616.6312000006);
        assert_eq!(geo_poly.rings[0][316].y, 1081003.3420000002);
        let mut grid = RasterGrid::from_polygon(&geo_poly, 5.0);
        // grid.rasterize(&geo_poly);
        // grid.print();
        grid.add_padding(50.0).expect("Failed to add padding");
        grid.print();

        let geo_polys = read_shapefile_polygons(path).expect("Failed to read shapefile polygons");
        println!("number of polygons read: {}", geo_polys.len());
    }
}
