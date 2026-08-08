use data_processor::caaml_parser::*;

#[test_log::test]
fn test_parse_bulletins() {
    let json_data = include_str!("fixtures/caaml_sample.json");
    match parse_avalanche_bulletins(json_data) {
        Ok(parsed) => {
            println!("Successfully parsed collection!");
            if let Some(bulletin) = parsed.bulletins.first() {
                println!("Bulletin ID: {:?}", bulletin.bulletin_id);
                if let Some(ratings) = &bulletin.danger_ratings {
                    println!("Main Danger Rating: {:?}", ratings[0].main_value);
                }
            }
        }
        Err(e) => eprintln!("Failed to parse JSON: {}", e),
    }
}

#[test]
fn test_parse_minimal_valid_bulletin() {
    let json = r#"
        {
            "bulletins": [
                {
                    "bulletinID": "BULLETIN-MIN-001"
                }
            ]
        }
        "#;

    let parsed = parse_avalanche_bulletins(json).expect("Should parse minimal JSON");
    assert_eq!(parsed.bulletins.len(), 1);
    assert_eq!(
        parsed.bulletins[0].bulletin_id.as_deref(),
        Some("BULLETIN-MIN-001")
    );
}

#[test]
fn test_parse_full_bulletin() {
    let json = r#"
        {
            "bulletins": [
                {
                    "bulletinID": "BULLETIN-FULL-2026",
                    "lang": "en",
                    "publicationTime": "2026-03-30T07:30:00Z",
                    "nextUpdate": "2026-03-31T07:30:00Z",
                    "unscheduled": false,
                    "highlights": "Severe wind slabs at high elevations.",
                    "validTime": {
                        "startTime": "2026-03-30T08:00:00Z",
                        "endTime": "2026-03-31T08:00:00Z"
                    },
                    "source": {
                        "provider": {
                            "name": "Avalanche Warning Service",
                            "website": "https://example.com/aws",
                            "contactPerson": {
                                "name": "Jane Doe"
                            }
                        }
                    },
                    "regions": [
                        {
                            "regionID": "AT-07-01",
                            "name": "Tyrol North"
                        }
                    ],
                    "dangerRatings": [
                        {
                            "mainValue": "high",
                            "elevation": {
                                "lowerBound": "2000"
                            },
                            "aspects": ["N", "NE", "NW", "n/a"],
                            "validTimePeriod": "all_day"
                        }
                    ],
                    "avalancheProblems": [
                        {
                            "problemType": "wind_slab",
                            "comment": "Fresh wind drifts on shaded slopes.",
                            "avalancheSize": 3,
                            "snowpackStability": "poor",
                            "frequency": "many",
                            "dangerRatingValue": "high",
                            "validTimePeriod": "earlier"
                        }
                    ],
                    "weatherForecast": {
                        "highlights": "Sunny day",
                        "comment": "Temperatures rising up to +5°C."
                    },
                    "customData": {
                        "aws_specific_id": 98765,
                        "flag": true
                    }
                }
            ]
        }
        "#;

    let parsed = parse_avalanche_bulletins(json).expect("Should parse full JSON");
    let bulletin = &parsed.bulletins[0];

    assert_eq!(bulletin.bulletin_id.as_deref(), Some("BULLETIN-FULL-2026"));
    assert_eq!(bulletin.unscheduled, Some(false));

    // Source verification
    let source = bulletin.source.as_ref().unwrap();
    let provider = source.provider.as_ref().unwrap();
    assert_eq!(provider.name.as_deref(), Some("Avalanche Warning Service"));
    assert_eq!(
        provider.contact_person.as_ref().unwrap().name.as_deref(),
        Some("Jane Doe")
    );

    // Danger Rating verification
    let danger_ratings = bulletin.danger_ratings.as_ref().unwrap();
    assert_eq!(danger_ratings[0].main_value, DangerRatingValue::High);
    assert_eq!(
        danger_ratings[0].valid_time_period,
        Some(ValidTimePeriod::AllDay)
    );

    // Aspect verification (including the "n/a" variant)
    let aspects = danger_ratings[0].aspects.as_ref().unwrap();
    assert_eq!(
        aspects,
        &vec![Aspect::N, Aspect::NE, Aspect::NW, Aspect::NotApplicable]
    );

    // Avalanche Problem verification
    let problems = bulletin.avalanche_problems.as_ref().unwrap();
    assert_eq!(problems[0].problem_type, AvalancheProblemType::WindSlab);
    assert_eq!(problems[0].avalanche_size, Some(3));
    assert_eq!(
        problems[0].snowpack_stability,
        Some(SnowpackStability::Poor)
    );
    assert_eq!(problems[0].frequency, Some(Frequency::Many));

    // Custom Data dynamic payload verification
    let custom_data = bulletin.custom_data.as_ref().unwrap();
    assert_eq!(custom_data["aws_specific_id"], 98765);
}

#[test]
fn test_tendency_untagged_enum() {
    let json = r#"
        {
            "bulletins": [
                {
                    "bulletinID": "BULLETIN-TENDENCY",
                    "tendency": [
                        {
                            "highlights": "Improving situation",
                            "comment": "Danger will decrease tomorrow."
                        },
                        {
                            "tendencyType": "decreasing",
                            "validTime": {
                                "startTime": "2026-03-31T08:00:00Z",
                                "endTime": "2026-04-01T08:00:00Z"
                            }
                        }
                    ]
                }
            ]
        }
        "#;

    let parsed = parse_avalanche_bulletins(json).expect("Should parse untagged tendency list");
    let tendency_list = parsed.bulletins[0].tendency.as_ref().unwrap();

    assert_eq!(tendency_list.len(), 2);

    // First item should match Texts
    match &tendency_list[0] {
        TendencyItem::Texts(texts) => {
            assert_eq!(texts.highlights.as_deref(), Some("Improving situation"));
        }
        _ => panic!("Expected TendencyItem::Texts"),
    }
}

#[test]
fn test_error_missing_required_root_field() {
    let json = r#"
        {
            "metaData": { "comment": "Root level comment" }
        }
        "#;

    let result = parse_avalanche_bulletins(json);
    assert!(
        result.is_err(),
        "Should fail because 'bulletins' is missing"
    );
}

#[test]
fn test_error_invalid_enum_variant() {
    let json = r#"
        {
            "bulletins": [
                {
                    "dangerRatings": [
                        {
                            "mainValue": "extreme_catastrophe"
                        }
                    ]
                }
            ]
        }
        "#;

    let result = parse_avalanche_bulletins(json);
    assert!(
        result.is_err(),
        "Should fail due to unmapped enum string 'extreme_catastrophe'"
    );
}

#[test]
fn test_error_missing_required_inner_field() {
    let json = r#"
        {
            "bulletins": [
                {
                    "regions": [
                        {
                            "name": "Region without ID"
                        }
                    ]
                }
            ]
        }
        "#;

    let result = parse_avalanche_bulletins(json);
    assert!(
        result.is_err(),
        "Should fail because 'regionID' is required in region"
    );
}

#[test]
fn test_serialization_roundtrip() {
    let original_struct = AvalancheBulletinCollection {
        bulletins: vec![Bulletin {
            bulletin_id: Some("ROUNDTRIP-01".to_string()),
            lang: Some("de".to_string()),
            publication_time: None,
            valid_time: None,
            next_update: None,
            unscheduled: Some(true),
            source: None,
            regions: Some(vec![Region {
                region_id: "AT-08".to_string(),
                name: Some("Vorarlberg".to_string()),
                meta_data: None,
                custom_data: None,
            }]),
            danger_ratings: Some(vec![DangerRating {
                main_value: DangerRatingValue::Moderate,
                elevation: Some(Elevation {
                    lower_bound: Some("treeline".to_string()),
                    upper_bound: None,
                }),
                aspects: Some(vec![Aspect::E, Aspect::SE, Aspect::S]),
                valid_time_period: Some(ValidTimePeriod::Later),
                meta_data: None,
                custom_data: None,
            }]),
            avalanche_problems: None,
            highlights: None,
            weather_forecast: None,
            weather_review: None,
            avalanche_activity: None,
            snowpack_structure: None,
            travel_advisory: None,
            tendency: None,
            meta_data: None,
            custom_data: None,
        }],
        meta_data: None,
        custom_data: None,
    };

    // Serialize struct to JSON string
    let json_string = serde_json::to_string(&original_struct).expect("Serialization failed");

    // Deserialize string back to struct
    let deserialized_struct: AvalancheBulletinCollection =
        parse_avalanche_bulletins(&json_string).expect("Deserialization failed");

    // Verify equality
    assert_eq!(
        deserialized_struct.bulletins[0].bulletin_id,
        original_struct.bulletins[0].bulletin_id
    );
    assert_eq!(
        deserialized_struct.bulletins[0].regions.as_ref().unwrap()[0].region_id,
        "AT-08"
    );
    assert_eq!(
        deserialized_struct.bulletins[0]
            .danger_ratings
            .as_ref()
            .unwrap()[0]
            .main_value,
        DangerRatingValue::Moderate
    );
}

fn sample_collection() -> AvalancheBulletinCollection {
    let json = r#"
        {
            "bulletins": [
                {
                    "bulletinID": "BULLETIN-NORTH",
                    "regions": [
                        { "regionID": "REG-01", "name": "Northern Alps" }
                    ]
                },
                {
                    "bulletinID": "BULLETIN-SOUTH",
                    "regions": [
                        { "regionID": "REG-02", "name": "Southern Alps" }
                    ]
                }
            ]
        }
        "#;
    parse_avalanche_bulletins(json).unwrap()
}

#[test]
fn test_lookup_by_region_id() {
    let collection = sample_collection();
    let bulletin = collection.get_bulletin_by_region("REG-01");
    assert!(bulletin.is_some());
    assert_eq!(
        bulletin.unwrap().bulletin_id.as_deref(),
        Some("BULLETIN-NORTH")
    );
}

#[test]
fn test_lookup_by_region_name_case_insensitive() {
    let collection = sample_collection();
    let bulletin = collection.get_bulletin_by_region("southern alps");
    assert!(bulletin.is_some());
    assert_eq!(
        bulletin.unwrap().bulletin_id.as_deref(),
        Some("BULLETIN-SOUTH")
    );
}

#[test]
fn test_lookup_not_found() {
    let collection = sample_collection();
    assert!(collection.get_bulletin_by_region("REG-99").is_none());
}

#[test]
fn test_print_summary() {
    let raw_json = r#"
    {
      "bulletins": [
        {
          "bulletinID": "BULLETIN-2026-0330",
          "lang": "en",
          "publicationTime": "2026-03-30T07:30:00Z",
          "validTime": {
            "startTime": "2026-03-30T08:00:00Z",
            "endTime": "2026-03-31T08:00:00Z"
          },
          "highlights": "Critical avalanche situation due to fresh wind slabs!",
          "regions": [
            { "regionID": "AT-07-01", "name": "Northern Alps" },
            { "regionID": "AT-07-02", "name": "Kitzbühel Alps" }
          ],
          "dangerRatings": [
            {
              "mainValue": "high",
              "validTimePeriod": "all_day",
              "elevation": { "lowerBound": "2000" },
              "aspects": ["N", "NE", "NW"]
            }
          ],
          "avalancheProblems": [
            {
              "problemType": "wind_slab",
              "avalancheSize": 3,
              "snowpackStability": "poor",
              "frequency": "many",
              "comment": "Fresh wind drifts formed near ridges are extremely trigger-sensitive."
            }
          ],
          "snowpackStructure": {
            "comment": "Weak layers exist within the upper snowpack above 2000m."
          }
        }
      ]
    }
    "#;

    let collection = parse_avalanche_bulletins(raw_json).expect("Failed to parse bulletin JSON");

    if let Some(bulletin) = collection.bulletins.first() {
        bulletin.print_summary();
    }
}

#[test]
fn test_print_summary_with_sample_json() {
    let json_data = include_str!("fixtures/caaml_sample.json");
    let collection =
        parse_avalanche_bulletins(json_data).expect("Failed to parse sample bulletin JSON");

    if let Some(bulletin) = collection.get_bulletin_by_region("Kalkkögel") {
        bulletin.print_summary();
        assert_eq!(
            bulletin
                .danger_ratings
                .clone()
                .expect("Expected danger ratings")
                .first()
                .unwrap()
                .main_value,
            DangerRatingValue::Low
        );
        assert!(
            bulletin
                .danger_ratings
                .clone()
                .expect("Expected danger ratings")
                .first()
                .unwrap()
                .elevation
                .clone()
                .unwrap()
                .lower_bound
                .is_none()
        );
        assert_eq!(
            bulletin
                .danger_ratings
                .clone()
                .expect("Expected danger ratings")
                .first()
                .unwrap()
                .elevation
                .clone()
                .unwrap()
                .upper_bound
                .unwrap(),
            "treeline"
        );
        assert_eq!(
            bulletin
                .danger_ratings
                .clone()
                .expect("Expected danger ratings")
                .last()
                .unwrap()
                .main_value,
            DangerRatingValue::Moderate
        );
        assert_eq!(
            bulletin
                .danger_ratings
                .clone()
                .expect("Expected danger ratings")
                .last()
                .unwrap()
                .elevation
                .clone()
                .unwrap()
                .lower_bound
                .unwrap(),
            "treeline"
        );
        assert!(
            bulletin
                .danger_ratings
                .clone()
                .expect("Expected danger ratings")
                .last()
                .unwrap()
                .elevation
                .clone()
                .unwrap()
                .upper_bound
                .is_none()
        );
        assert_eq!(
            bulletin
                .avalanche_problems
                .clone()
                .expect("Expected avalanche problems")
                .first()
                .unwrap()
                .problem_type,
            AvalancheProblemType::WindSlab
        );
        assert_eq!(
            bulletin
                .avalanche_problems
                .clone()
                .expect("Expected avalanche problems")
                .first()
                .unwrap()
                .avalanche_size
                .unwrap(),
            2
        );
        assert_eq!(
            bulletin
                .avalanche_problems
                .clone()
                .expect("Expected avalanche problems")
                .first()
                .unwrap()
                .frequency
                .clone()
                .unwrap(),
            Frequency::Some
        );
        assert_eq!(
            bulletin
                .avalanche_problems
                .clone()
                .expect("Expected avalanche problems")
                .first()
                .unwrap()
                .snowpack_stability
                .clone()
                .unwrap(),
            SnowpackStability::Poor
        );
        assert_eq!(
            bulletin
                .avalanche_problems
                .clone()
                .expect("Expected avalanche problems")
                .first()
                .unwrap()
                .aspects,
            Some(vec![Aspect::NW, Aspect::NE, Aspect::N, Aspect::E])
        );
    }
}

#[test]
fn test_analyze_bulletin() {
    let json_data = include_str!("fixtures/caaml_sample.json");
    let collection =
        parse_avalanche_bulletins(json_data).expect("Failed to parse sample bulletin JSON");

    if let Some(bulletin) = collection.get_bulletin_by_region("Kalkkögel") {
        let analysis = bulletin.analyze();
        println!("Analysis: {:?}", analysis);
        assert_eq!(analysis.danger_ratings.len(), 2);
        assert_eq!(analysis.target_problems.len(), 1);
        assert_eq!(analysis.max_avalanche_size, 2);

        assert_eq!(analysis.danger_ratings[0].lower_elevation_m, 0); // Missing lower -> 0
        assert_eq!(analysis.danger_ratings[0].upper_elevation_m, 1800); // "treeline" -> 1800
        assert_eq!(analysis.danger_ratings[1].lower_elevation_m, 1800); // "treeline" -> 1800
        assert_eq!(analysis.danger_ratings[1].upper_elevation_m, 9000); // Missing upper -> 9000
        assert_eq!(
            analysis.combined_aspect_mask,
            aspects_to_bitmask(&[Aspect::NW, Aspect::NE, Aspect::N, Aspect::E])
        );
        assert_eq!(analysis.combined_aspect_mask, 135);
    }
}

#[cfg(test)]
mod danger_rating_numeric_tests {
    use super::*;

    #[test]
    fn test_danger_rating_to_numeric() {
        assert_eq!(DangerRatingValue::Low.to_numeric(), 1);
        assert_eq!(DangerRatingValue::Moderate.to_numeric(), 2);
        assert_eq!(DangerRatingValue::Considerable.to_numeric(), 3);
        assert_eq!(DangerRatingValue::High.to_numeric(), 4);
        assert_eq!(DangerRatingValue::VeryHigh.to_numeric(), 5);
        assert_eq!(DangerRatingValue::NoSnow.to_numeric(), 0);
        assert_eq!(DangerRatingValue::NoRating.to_numeric(), 0);
    }

    #[test]
    fn test_danger_rating_from_numeric_valid() {
        assert_eq!(
            DangerRatingValue::from_numeric(1),
            Some(DangerRatingValue::Low)
        );
        assert_eq!(
            DangerRatingValue::from_numeric(2),
            Some(DangerRatingValue::Moderate)
        );
        assert_eq!(
            DangerRatingValue::from_numeric(3),
            Some(DangerRatingValue::Considerable)
        );
        assert_eq!(
            DangerRatingValue::from_numeric(4),
            Some(DangerRatingValue::High)
        );
        assert_eq!(
            DangerRatingValue::from_numeric(5),
            Some(DangerRatingValue::VeryHigh)
        );
        assert_eq!(
            DangerRatingValue::from_numeric(0),
            Some(DangerRatingValue::NoSnow)
        );
    }

    #[test]
    fn test_danger_rating_from_numeric_invalid() {
        assert_eq!(DangerRatingValue::from_numeric(6), None);
        assert_eq!(DangerRatingValue::from_numeric(99), None);
        assert_eq!(DangerRatingValue::from_numeric(u8::MAX), None);
    }

    #[test]
    fn test_numeric_roundtrip() {
        let variants = vec![
            DangerRatingValue::Low,
            DangerRatingValue::Moderate,
            DangerRatingValue::Considerable,
            DangerRatingValue::High,
            DangerRatingValue::VeryHigh,
            DangerRatingValue::NoSnow,
            // Note: NoRating maps to 0 as well, so roundtripping 0 returns NoSnow based on your implementation
        ];

        for variant in variants {
            let num = variant.to_numeric();
            let back = DangerRatingValue::from_numeric(num).unwrap();
            assert_eq!(variant, back);
        }
    }
}
#[cfg(test)]
mod aspect_bitmask_tests {
    use super::*;

    #[test]
    fn test_aspect_to_bitflag() {
        assert_eq!(aspect_to_bitflag(&Aspect::N), 1); // 1 << 0
        assert_eq!(aspect_to_bitflag(&Aspect::NE), 2); // 1 << 1
        assert_eq!(aspect_to_bitflag(&Aspect::E), 4); // 1 << 2
        assert_eq!(aspect_to_bitflag(&Aspect::SE), 8); // 1 << 3
        assert_eq!(aspect_to_bitflag(&Aspect::S), 16); // 1 << 4
        assert_eq!(aspect_to_bitflag(&Aspect::SW), 32); // 1 << 5
        assert_eq!(aspect_to_bitflag(&Aspect::W), 64); // 1 << 6
        assert_eq!(aspect_to_bitflag(&Aspect::NW), 128); // 1 << 7
        assert_eq!(aspect_to_bitflag(&Aspect::NotApplicable), 0);
    }

    #[test]
    fn test_aspects_to_bitmask_combinations() {
        // Empty list should yield 0
        assert_eq!(aspects_to_bitmask(&[]), 0);

        // Single aspect
        assert_eq!(aspects_to_bitmask(&[Aspect::N]), 1);

        // Multiple aspects (e.g., N, NE, NW -> 1 | 2 | 128 = 131)
        let aspects = vec![Aspect::N, Aspect::NE, Aspect::NW];
        assert_eq!(aspects_to_bitmask(&aspects), 131);

        // All cardinal and intercardinal aspects combined -> 255 (0b1111_1111)
        let all_aspects = vec![
            Aspect::N,
            Aspect::NE,
            Aspect::E,
            Aspect::SE,
            Aspect::S,
            Aspect::SW,
            Aspect::W,
            Aspect::NW,
        ];
        assert_eq!(aspects_to_bitmask(&all_aspects), 255);

        // Including NotApplicable shouldn't alter the bitmask
        let mixed = vec![Aspect::N, Aspect::NotApplicable];
        assert_eq!(aspects_to_bitmask(&mixed), 1);
    }

    #[test]
    fn test_bitmask_to_aspect_names() {
        // Zero mask should return empty list
        assert!(bitmask_to_aspect_names(0).is_empty());

        // Single mask (4 -> E)
        assert_eq!(bitmask_to_aspect_names(4), vec!["E"]);

        // Combined mask (131 -> N, NE, NW)
        let names = bitmask_to_aspect_names(131);
        assert_eq!(names, vec!["N", "NE", "NW"]);

        // Full mask (255 -> all 8 directions)
        let all_names = bitmask_to_aspect_names(255);
        assert_eq!(all_names, vec!["N", "NE", "E", "SE", "S", "SW", "W", "NW"]);
    }

    #[test]
    fn test_bitmask_roundtrip() {
        // Convert a slice to bitmask, then convert the bitmask back to names and verify consistency
        let original = vec![Aspect::N, Aspect::SE, Aspect::W];
        let mask = aspects_to_bitmask(&original);
        let names = bitmask_to_aspect_names(mask);

        assert_eq!(names, vec!["N", "SE", "W"]);
    }
}
