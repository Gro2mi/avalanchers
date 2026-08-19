//! CAAML avalanche-bulletin parsing: `AvalancheBulletinCollection` / `Bulletin`, `DangerRating` (with numeric conversion) and `AvalancheProblem`, plus human-readable summaries.
use serde::{Deserialize, Serialize};
use std::fmt::Write;

/// Root object for the CAAMLv6 collection
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AvalancheBulletinCollection {
    pub bulletins: Vec<Bulletin>,
    pub meta_data: Option<MetaData>,
    pub custom_data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bulletin {
    #[serde(rename = "bulletinID")]
    pub bulletin_id: Option<String>,
    pub lang: Option<String>,
    pub publication_time: Option<String>,
    pub valid_time: Option<ValidTime>,
    pub next_update: Option<String>,
    pub unscheduled: Option<bool>,
    pub source: Option<Source>,
    pub regions: Option<Vec<Region>>,
    pub danger_ratings: Option<Vec<DangerRating>>,
    pub avalanche_problems: Option<Vec<AvalancheProblem>>,
    pub highlights: Option<String>,
    pub weather_forecast: Option<Texts>,
    pub weather_review: Option<Texts>,
    pub avalanche_activity: Option<Texts>,
    pub snowpack_structure: Option<Texts>,
    pub travel_advisory: Option<Texts>,
    pub tendency: Option<Vec<TendencyItem>>,
    pub meta_data: Option<MetaData>,
    pub custom_data: Option<serde_json::Value>,
}

impl Bulletin {
    /// Generates a human-readable summary string of the avalanche bulletin.
    pub fn summary(&self) -> String {
        let mut out = String::new();

        writeln!(out, "==================================================").unwrap();
        writeln!(out, "             AVALANCHE BULLETIN SUMMARY           ").unwrap();
        writeln!(out, "==================================================").unwrap();

        // Metadata
        if let Some(id) = &self.bulletin_id {
            writeln!(out, "ID:           {}", id).unwrap();
        }
        if let Some(lang) = &self.lang {
            writeln!(out, "Language:     {}", lang).unwrap();
        }
        if let Some(pub_time) = &self.publication_time {
            writeln!(out, "Published:    {}", pub_time).unwrap();
        }
        if let Some(valid) = &self.valid_time {
            let start = valid.start_time.as_deref().unwrap_or("N/A");
            let end = valid.end_time.as_deref().unwrap_or("N/A");
            writeln!(out, "Valid Period: {} -> {}", start, end).unwrap();
        }

        // Highlights
        if let Some(highlights) = &self.highlights {
            writeln!(out, "\n--- HIGHLIGHTS ---").unwrap();
            writeln!(out, "{}", highlights).unwrap();
        }

        // 3. Affected Regions
        if let Some(regions) = &self.regions {
            writeln!(out, "\n--- REGIONS ---").unwrap();
            for region in regions {
                let name = region.name.as_deref().unwrap_or("Unnamed Region");
                writeln!(out, " • {} (ID: {})", name, region.region_id).unwrap();
            }
        }

        // Danger Ratings
        if let Some(ratings) = &self.danger_ratings {
            writeln!(out, "\n--- DANGER RATINGS ---").unwrap();
            for (i, rating) in ratings.iter().enumerate() {
                write!(out, " {}. Level: {:?}", i + 1, rating.main_value).unwrap();
                if let Some(period) = &rating.valid_time_period {
                    write!(out, " ({:?})", period).unwrap();
                }
                writeln!(out).unwrap();

                if let Some(elev) = &rating.elevation {
                    let lower = elev.lower_bound.as_deref().unwrap_or("surface");
                    let upper = elev.upper_bound.as_deref().unwrap_or("unlimited");
                    writeln!(out, "    Elevation: {} to {}", lower, upper).unwrap();
                }

                if let Some(aspects) = &rating.aspects {
                    let aspect_list: Vec<String> =
                        aspects.iter().map(|a| format!("{:?}", a)).collect();
                    writeln!(out, "    Aspects:   {}", aspect_list.join(", ")).unwrap();
                }
            }
        }

        // Avalanche Problems
        if let Some(problems) = &self.avalanche_problems {
            writeln!(out, "\n--- AVALANCHE PROBLEMS ---").unwrap();
            for (i, prob) in problems.iter().enumerate() {
                writeln!(out, " {}. Type: {:?}", i + 1, prob.problem_type).unwrap();
                if let Some(size) = prob.avalanche_size {
                    writeln!(out, "    - Expected Size:     {}", size).unwrap();
                }
                if let Some(stability) = &prob.snowpack_stability {
                    writeln!(out, "    - Snow Stability:    {:?}", stability).unwrap();
                }
                if let Some(freq) = &prob.frequency {
                    writeln!(out, "    - Frequency:         {:?}", freq).unwrap();
                }
                if let Some(comment) = &prob.comment {
                    writeln!(out, "    - Details:           {}", comment).unwrap();
                }
            }
        }

        // 6. Text Synopses (Snowpack, Weather, Advisory)
        if let Some(snow) = &self.snowpack_structure
            && let Some(comment) = &snow.comment
        {
            writeln!(out, "\n--- SNOWPACK STRUCTURE ---").unwrap();
            writeln!(out, "{}", comment).unwrap();
        }

        if let Some(advisory) = &self.travel_advisory
            && let Some(comment) = &advisory.comment
        {
            writeln!(out, "\n--- TRAVEL ADVISORY ---").unwrap();
            writeln!(out, "{}", comment).unwrap();
        }

        writeln!(out, "==================================================").unwrap();
        out
    }

    pub fn print_summary(&self) {
        println!("{}", self.summary());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TendencyItem {
    Texts(Texts),
    Tendency(Tendency),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DangerRating {
    pub main_value: DangerRatingValue,
    pub elevation: Option<Elevation>,
    pub aspects: Option<Vec<Aspect>>,
    pub valid_time_period: Option<ValidTimePeriod>,
    pub meta_data: Option<MetaData>,
    pub custom_data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DangerRatingValue {
    Low,
    Moderate,
    Considerable,
    High,
    VeryHigh,
    NoSnow,
    NoRating,
}

impl DangerRatingValue {
    pub fn to_numeric(&self) -> u8 {
        match self {
            DangerRatingValue::Low => 1,
            DangerRatingValue::Moderate => 2,
            DangerRatingValue::Considerable => 3,
            DangerRatingValue::High => 4,
            DangerRatingValue::VeryHigh => 5,
            DangerRatingValue::NoSnow => 0,
            DangerRatingValue::NoRating => 0,
        }
    }

    pub fn from_numeric(value: u8) -> Option<Self> {
        match value {
            1 => Some(DangerRatingValue::Low),
            2 => Some(DangerRatingValue::Moderate),
            3 => Some(DangerRatingValue::Considerable),
            4 => Some(DangerRatingValue::High),
            5 => Some(DangerRatingValue::VeryHigh),
            0 => Some(DangerRatingValue::NoSnow), // or NoRating, depending on context
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AvalancheProblem {
    pub problem_type: AvalancheProblemType,
    pub comment: Option<String>,
    pub avalanche_size: Option<u8>,
    pub snowpack_stability: Option<SnowpackStability>,
    pub frequency: Option<Frequency>,
    pub danger_rating_value: Option<DangerRatingValue>,
    pub elevation: Option<Elevation>,
    pub aspects: Option<Vec<Aspect>>,
    pub valid_time_period: Option<ValidTimePeriod>,
    pub meta_data: Option<MetaData>,
    pub custom_data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AvalancheProblemType {
    NewSnow,
    WindSlab,
    PersistentWeakLayers,
    WetSnow,
    GlidingSnow,
    Cornices,
    NoDistinctAvalancheProblem,
    FavourableSituation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SnowpackStability {
    Good,
    Fair,
    Poor,
    VeryPoor,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Frequency {
    None,
    Few,
    Some,
    Many,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tendency {
    pub tendency_type: Option<TendencyType>,
    pub valid_time: Option<ValidTime>,
    pub meta_data: Option<MetaData>,
    pub custom_data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TendencyType {
    Decreasing,
    Steady,
    Increasing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetaData {
    pub ext_files: Option<Vec<ExtFile>>,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtFile {
    pub file_type: Option<String>,
    pub description: Option<String>,
    #[serde(rename = "fileReferenceURI")]
    pub file_reference_uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    pub provider: Option<Provider>,
    pub person: Option<Person>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Provider {
    pub name: Option<String>,
    pub website: Option<String>,
    pub contact_person: Option<Person>,
    pub meta_data: Option<MetaData>,
    pub custom_data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Person {
    pub name: Option<String>,
    pub website: Option<String>,
    pub meta_data: Option<MetaData>,
    pub custom_data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Region {
    #[serde(rename = "regionID")]
    pub region_id: String,
    pub name: Option<String>,
    pub meta_data: Option<MetaData>,
    pub custom_data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Aspect {
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
    #[serde(rename = "n/a")]
    NotApplicable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Elevation {
    pub lower_bound: Option<String>,
    pub upper_bound: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidTime {
    pub start_time: Option<String>,
    pub end_time: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidTimePeriod {
    AllDay,
    Earlier,
    Later,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Texts {
    pub highlights: Option<String>,
    pub comment: Option<String>,
}

/// Helper function to parse JSON input string into the Rust struct collection
pub fn parse_avalanche_bulletins(
    json_data: &str,
) -> Result<AvalancheBulletinCollection, serde_json::Error> {
    serde_json::from_str(json_data)
}

impl AvalancheBulletinCollection {
    /// Returns the first `Bulletin` matching the given query string by either
    /// `region_id` or region `name` (case-insensitive).
    pub fn get_bulletin_by_region(&self, query: &str) -> Option<&Bulletin> {
        let query_lower = query.to_lowercase();

        self.bulletins.iter().find(|bulletin| {
            bulletin.regions.as_ref().is_some_and(|regions| {
                regions.iter().any(|region| {
                    let matches_id = region.region_id.to_lowercase() == query_lower;
                    let matches_name = region
                        .name
                        .as_deref()
                        .is_some_and(|name| name.to_lowercase() == query_lower);

                    matches_id || matches_name
                })
            })
        })
    }
}

/// Standard average treeline estimate in the Austrian Alps (~1800m)
pub const AUSTRIAN_TREELINE_METERS: u32 = 1800;

// =========================================================================
// Aspect Bitmask Encoding (1 byte / u8)
// =========================================================================
// N  = Bit 0 (0b0000_0001) = 1
// NE = Bit 1 (0b0000_0010) = 2
// E  = Bit 2 (0b0000_0100) = 4
// SE = Bit 3 (0b0000_1000) = 8
// S  = Bit 4 (0b0001_0000) = 16
// SW = Bit 5 (0b0010_0000) = 32
// W  = Bit 6 (0b0100_0000) = 64
// NW = Bit 7 (0b1000_0000) = 128

pub fn aspect_to_bitflag(aspect: &Aspect) -> u8 {
    match aspect {
        Aspect::N => 1 << 0,
        Aspect::NE => 1 << 1,
        Aspect::E => 1 << 2,
        Aspect::SE => 1 << 3,
        Aspect::S => 1 << 4,
        Aspect::SW => 1 << 5,
        Aspect::W => 1 << 6,
        Aspect::NW => 1 << 7,
        Aspect::NotApplicable => 0,
    }
}

pub fn aspects_to_bitmask(aspects: &[Aspect]) -> u8 {
    aspects
        .iter()
        .fold(0u8, |acc, aspect| acc | aspect_to_bitflag(aspect))
}

/// Helper function to format bitmask as readable string list (e.g. "N, NE, NW")
pub fn bitmask_to_aspect_names(mask: u8) -> Vec<&'static str> {
    let mut vec = Vec::new();
    if mask & (1 << 0) != 0 {
        vec.push("N");
    }
    if mask & (1 << 1) != 0 {
        vec.push("NE");
    }
    if mask & (1 << 2) != 0 {
        vec.push("E");
    }
    if mask & (1 << 3) != 0 {
        vec.push("SE");
    }
    if mask & (1 << 4) != 0 {
        vec.push("S");
    }
    if mask & (1 << 5) != 0 {
        vec.push("SW");
    }
    if mask & (1 << 6) != 0 {
        vec.push("W");
    }
    if mask & (1 << 7) != 0 {
        vec.push("NW");
    }
    vec
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessedDangerRating {
    pub main_value: DangerRatingValue,
    pub lower_elevation_m: u32,
    pub upper_elevation_m: u32,
    pub aspect_mask: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TargetProblemSummary {
    pub problem_type: AvalancheProblemType,
    pub avalanche_size: u8,
    pub aspect_mask: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BulletinAnalysis {
    pub danger_ratings: Vec<ProcessedDangerRating>,
    /// Individual filtered problems (NewSnow, WindSlab, PersistentWeakLayers)
    pub target_problems: Vec<TargetProblemSummary>,
    /// Highest avalanche size across ALL target problem types
    pub max_avalanche_size: u8,
    /// Set of ALL affected aspects across target problem types encoded as a u8 bitmask
    pub combined_aspect_mask: u8,
}

fn parse_elevation_bound(bound: Option<&str>, default_value: u32) -> u32 {
    match bound {
        Some(s) if s.eq_ignore_ascii_case("treeline") => AUSTRIAN_TREELINE_METERS,
        Some(s) => s.parse::<u32>().unwrap_or(default_value),
        None => default_value,
    }
}

impl Bulletin {
    pub fn analyze(&self) -> BulletinAnalysis {
        // Process Danger Ratings
        let danger_ratings = self
            .danger_ratings
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|dr| {
                let (lower_str, upper_str) = match &dr.elevation {
                    Some(e) => (e.lower_bound.as_deref(), e.upper_bound.as_deref()),
                    None => (None, None),
                };

                let lower_elevation_m = parse_elevation_bound(lower_str, 0);
                let upper_elevation_m = parse_elevation_bound(upper_str, 9000);
                let aspect_mask = dr.aspects.as_deref().map(aspects_to_bitmask).unwrap_or(0);

                ProcessedDangerRating {
                    main_value: dr.main_value.clone(),
                    lower_elevation_m,
                    upper_elevation_m,
                    aspect_mask,
                }
            })
            .collect();

        // Filter Avalanche Problems for NewSnow, WindSlab, PersistentWeakLayers
        let target_problems: Vec<TargetProblemSummary> = self
            .avalanche_problems
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .filter(|p| {
                matches!(
                    p.problem_type,
                    AvalancheProblemType::NewSnow
                        | AvalancheProblemType::WindSlab
                        | AvalancheProblemType::PersistentWeakLayers
                )
            })
            .map(|p| {
                let aspect_mask = p.aspects.as_deref().map(aspects_to_bitmask).unwrap_or(0);

                TargetProblemSummary {
                    problem_type: p.problem_type.clone(),
                    avalanche_size: p.avalanche_size.unwrap_or(0),
                    aspect_mask,
                }
            })
            .collect();

        // Compute overall max avalanche size and combined aspect bitmask for target types
        let mut max_avalanche_size: u8 = 0;
        let mut combined_aspect_mask: u8 = 0;

        for problem in &target_problems {
            combined_aspect_mask |= problem.aspect_mask;
            if problem.avalanche_size > max_avalanche_size {
                max_avalanche_size = problem.avalanche_size;
            }
        }

        BulletinAnalysis {
            danger_ratings,
            target_problems,
            max_avalanche_size,
            combined_aspect_mask,
        }
    }
}
