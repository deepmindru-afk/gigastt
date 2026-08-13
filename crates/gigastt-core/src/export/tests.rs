use super::*;

fn sample_words() -> Vec<WordInfo> {
    vec![
        WordInfo {
            word: "привет".to_string(),
            start: 0.0,
            end: 0.5,
            confidence: 0.98,
            speaker: Some(0),
        },
        WordInfo {
            word: "как".to_string(),
            start: 0.6,
            end: 0.9,
            confidence: 0.95,
            speaker: Some(0),
        },
        WordInfo {
            word: "дела".to_string(),
            start: 1.0,
            end: 1.4,
            confidence: 0.97,
            speaker: Some(1),
        },
    ]
}

fn sample_result() -> TranscribeResult {
    TranscribeResult {
        text: "привет как дела".to_string(),
        words: sample_words(),
        duration_s: 1.4,
        confidence: None,
    }
}

#[test]
fn test_to_txt() {
    let result = sample_result();
    assert_eq!(to_txt(&result), "привет как дела");
}

#[test]
fn test_to_json() {
    let result = sample_result();
    let json = to_json(&result);
    assert!(json.contains("привет как дела"));
    assert!(json.contains("\"duration\":1.4"));
}

#[test]
fn test_to_srt() {
    let words = sample_words();
    let srt = to_srt(&words, 80, 14);
    assert!(srt.contains("00:00:00,000 -->"));
    assert!(srt.contains("[SPEAKER_0] привет как"));
    assert!(srt.contains("[SPEAKER_1] дела"));
    assert!(srt.starts_with("1\n"));
}

#[test]
fn test_to_vtt() {
    let words = sample_words();
    let vtt = to_vtt(&words, 80, 14);
    assert!(vtt.starts_with("WEBVTT\n\n"));
    assert!(vtt.contains("00:00:00.000 -->"));
    assert!(vtt.contains("[SPEAKER_1] дела"));
}

#[test]
fn test_to_md() {
    let result = sample_result();
    let md = to_md(&result, true);
    assert!(md.contains("duration: 1.4"));
    assert!(md.contains("speakers: 2"));
    assert!(md.contains("привет как дела"));
    assert!(md.contains("| Word | Start | End |"));
}

#[test]
fn test_format_srt_time() {
    assert_eq!(format_srt_time(0.0), "00:00:00,000");
    assert_eq!(format_srt_time(61.123), "00:01:01,123");
    assert_eq!(format_srt_time(3661.5), "01:01:01,500");
}

#[test]
fn test_format_vtt_time() {
    assert_eq!(format_vtt_time(0.0), "00:00:00.000");
    assert_eq!(format_vtt_time(61.123), "00:01:01.123");
}

#[test]
fn test_export_format_from_str() {
    assert_eq!(ExportFormat::from_str("srt").unwrap(), ExportFormat::Srt);
    assert_eq!(ExportFormat::from_str("SRT").unwrap(), ExportFormat::Srt);
    assert_eq!(
        ExportFormat::from_str("markdown").unwrap(),
        ExportFormat::Md
    );
    assert!(ExportFormat::from_str("docx").is_err());
}

#[test]
fn test_empty_words() {
    let words: Vec<WordInfo> = Vec::new();
    assert!(to_srt(&words, 80, 14).is_empty());
    assert!(to_vtt(&words, 80, 14) == "WEBVTT\n\n");
}

#[test]
fn test_export_format_display_all_variants() {
    assert_eq!(ExportFormat::Json.to_string(), "json");
    assert_eq!(ExportFormat::Txt.to_string(), "txt");
    assert_eq!(ExportFormat::Srt.to_string(), "srt");
    assert_eq!(ExportFormat::Vtt.to_string(), "vtt");
    assert_eq!(ExportFormat::Md.to_string(), "md");
}

#[test]
fn test_export_format_content_type_all_variants() {
    assert_eq!(
        ExportFormat::Json.content_type(),
        "application/json; charset=utf-8"
    );
    assert_eq!(
        ExportFormat::Txt.content_type(),
        "text/plain; charset=utf-8"
    );
    assert_eq!(
        ExportFormat::Srt.content_type(),
        "application/x-subrip; charset=utf-8"
    );
    assert_eq!(ExportFormat::Vtt.content_type(), "text/vtt; charset=utf-8");
    assert_eq!(
        ExportFormat::Md.content_type(),
        "text/markdown; charset=utf-8"
    );
}

#[test]
fn test_export_format_extension_all_variants() {
    assert_eq!(ExportFormat::Json.extension(), "json");
    assert_eq!(ExportFormat::Txt.extension(), "txt");
    assert_eq!(ExportFormat::Srt.extension(), "srt");
    assert_eq!(ExportFormat::Vtt.extension(), "vtt");
    assert_eq!(ExportFormat::Md.extension(), "md");
}

#[test]
fn test_render_dispatches_each_format() {
    let result = sample_result();
    let opts = RenderOpts::default();

    let json = ExportFormat::Json.render(&result, &opts);
    assert_eq!(json, to_json(&result));

    let txt = ExportFormat::Txt.render(&result, &opts);
    assert_eq!(txt, "привет как дела");

    let srt = ExportFormat::Srt.render(&result, &opts);
    assert!(srt.starts_with("1\n"));

    let vtt = ExportFormat::Vtt.render(&result, &opts);
    assert!(vtt.starts_with("WEBVTT\n\n"));

    let md = ExportFormat::Md.render(&result, &opts);
    assert!(md.starts_with("---\n"));
    // Default opts disable word timestamps, so no table is emitted.
    assert!(!md.contains("| Word | Start | End |"));
}

#[test]
fn test_render_md_with_word_timestamps_opt_in() {
    let result = sample_result();
    let opts = RenderOpts {
        include_word_timestamps: true,
        ..RenderOpts::default()
    };
    let md = ExportFormat::Md.render(&result, &opts);
    assert!(md.contains("# Word timings"));
    assert!(md.contains("| Word | Start | End |"));
}

#[test]
fn test_render_opts_default_values() {
    let opts = RenderOpts::default();
    assert_eq!(opts.max_chars_per_line, 80);
    assert_eq!(opts.max_words_per_line, 14);
    assert!(!opts.include_word_timestamps);
}

#[test]
fn test_from_str_all_aliases() {
    assert_eq!(ExportFormat::from_str("json").unwrap(), ExportFormat::Json);
    assert_eq!(ExportFormat::from_str("txt").unwrap(), ExportFormat::Txt);
    assert_eq!(ExportFormat::from_str("text").unwrap(), ExportFormat::Txt);
    assert_eq!(ExportFormat::from_str("vtt").unwrap(), ExportFormat::Vtt);
    assert_eq!(ExportFormat::from_str("md").unwrap(), ExportFormat::Md);
}

#[test]
fn test_to_md_no_speakers_zero_count() {
    let result = TranscribeResult {
        text: "no speaker words".to_string(),
        words: vec![WordInfo {
            word: "no".to_string(),
            start: 0.0,
            end: 0.3,
            confidence: 0.9,
            speaker: None,
        }],
        duration_s: 0.3,
        confidence: None,
    };
    let md = to_md(&result, true);
    assert!(md.contains("speakers: 0"));
    // Speaker column renders "-" when no speaker is assigned.
    assert!(md.contains("| - |"));
}

#[test]
fn test_to_md_word_timestamps_skipped_when_empty() {
    let result = TranscribeResult {
        text: String::new(),
        words: Vec::new(),
        duration_s: 0.0,
        confidence: None,
    };
    let md = to_md(&result, true);
    // Empty word list means the appendix table is omitted entirely.
    assert!(!md.contains("# Word timings"));
    assert!(md.contains("speakers: 0"));
}

#[test]
fn test_to_md_escapes_pipe_in_word() {
    let result = TranscribeResult {
        text: "a|b".to_string(),
        words: vec![WordInfo {
            word: "a|b".to_string(),
            start: 0.0,
            end: 0.5,
            confidence: 0.91,
            speaker: Some(2),
        }],
        duration_s: 0.5,
        confidence: None,
    };
    let md = to_md(&result, true);
    // Pipe in the word must be escaped to avoid breaking the table column.
    assert!(md.contains("a\\|b"));
    assert!(md.contains("SPEAKER_2"));
    assert!(md.contains("speakers: 3"));
}

#[test]
fn test_srt_speaker_change_breaks_cue_with_label() {
    // Two speakers force a cue break; each cue carries its speaker label.
    let words = sample_words();
    let cues = build_cues(&words, 80, 14);
    assert_eq!(cues.len(), 2);
    assert!(cues[0].text.starts_with("[SPEAKER_0]"));
    assert!(cues[1].text.starts_with("[SPEAKER_1]"));
    assert!(cues[1].text.contains("дела"));
}

#[test]
fn test_line_breaking() {
    let words: Vec<WordInfo> = (0..20)
        .map(|i| WordInfo {
            word: format!("word{i}"),
            start: i as f64,
            end: i as f64 + 0.4,
            confidence: 0.9,
            speaker: None,
        })
        .collect();
    let srt = to_srt(&words, 40, 5);
    let cue_count = srt.trim().split("\n\n").count();
    // 20 words / 5 per line = 4 cues, but exact count depends on chars.
    assert!(cue_count >= 2);
}

#[test]
fn test_to_segments_shares_cue_boundaries() {
    // Two speakers force a cue break, so segments mirror the SRT cues:
    // one per speaker, with matching spans and per-segment word membership.
    let words = sample_words();
    let segments = to_segments(&words, 80, 14);
    assert_eq!(segments.len(), 2);

    assert_eq!(segments[0].start, 0.0);
    assert_eq!(segments[0].end, 0.9);
    assert!(segments[0].text.starts_with("[SPEAKER_0] привет"));
    assert_eq!(segments[0].words.len(), 2);
    assert_eq!(segments[0].words[0].word, "привет");
    assert_eq!(segments[0].words[1].word, "как");

    assert_eq!(segments[1].start, 1.0);
    assert_eq!(segments[1].end, 1.4);
    assert!(segments[1].text.contains("дела"));
    assert_eq!(segments[1].words.len(), 1);
    assert_eq!(segments[1].words[0].word, "дела");
}

#[test]
fn test_to_segments_word_cap_splits() {
    // A tight per-line cap groups the 20 words into multiple segments whose
    // spans and word membership line up with the flat list order.
    let words: Vec<WordInfo> = (0..20)
        .map(|i| WordInfo {
            word: format!("word{i}"),
            start: i as f64,
            end: i as f64 + 0.4,
            confidence: 0.9,
            speaker: None,
        })
        .collect();
    let segments = to_segments(&words, 0, 5);
    assert_eq!(segments.len(), 4);
    // Every word is accounted for exactly once, in order.
    let total: usize = segments.iter().map(|s| s.words.len()).sum();
    assert_eq!(total, 20);
    assert_eq!(segments[0].words[0].word, "word0");
    assert_eq!(segments[0].start, 0.0);
    assert_eq!(segments[0].end, 4.4);
    assert_eq!(segments[3].words.last().unwrap().word, "word19");
}

#[test]
fn test_to_segments_empty() {
    let words: Vec<WordInfo> = Vec::new();
    assert!(to_segments(&words, 80, 14).is_empty());
}

#[test]
fn test_to_segments_serializes_with_words() {
    let words = sample_words();
    let segments = to_segments(&words, 80, 14);
    let json = serde_json::to_value(&segments).unwrap();
    assert_eq!(json[0]["start"], 0.0);
    assert_eq!(json[0]["end"], 0.9);
    assert_eq!(json[0]["words"][0]["word"], "привет");
    // Speaker is carried through (skip_serializing_if only drops None).
    assert_eq!(json[0]["words"][0]["speaker"], 0);
}

#[test]
fn test_to_md_segments_emits_headers() {
    let result = sample_result();
    let md = to_md_segments(&result, 80, 14);
    // Frontmatter is preserved; the flat "# Transcript" blob is replaced by
    // per-segment "### [mm:ss]" headers.
    assert!(md.starts_with("---\n"));
    assert!(md.contains("duration: 1.4"));
    assert!(md.contains("speakers: 2"));
    assert!(md.contains("### [00:00]\n"));
    assert!(md.contains("### [00:01]\n"));
    assert!(md.contains("[SPEAKER_0] привет как"));
    assert!(md.contains("дела"));
    assert!(!md.contains("# Transcript"));
}

#[test]
fn test_to_md_segments_empty_words() {
    let result = TranscribeResult {
        text: String::new(),
        words: Vec::new(),
        duration_s: 0.0,
        confidence: None,
    };
    let md = to_md_segments(&result, 80, 14);
    // No words means no section headers, but the frontmatter still renders.
    assert!(md.starts_with("---\n"));
    assert!(md.contains("speakers: 0"));
    assert!(!md.contains("### ["));
}

#[test]
fn test_format_timestamp_hms() {
    // Under a minute, exactly a minute-plus, and past an hour widen as needed.
    assert_eq!(format_timestamp_hms(0.0), "00:00");
    assert_eq!(format_timestamp_hms(65.0), "01:05");
    assert_eq!(format_timestamp_hms(3661.0), "01:01:01");
    // Rounds to the nearest second; negatives clamp to zero.
    assert_eq!(format_timestamp_hms(59.6), "01:00");
    assert_eq!(format_timestamp_hms(-5.0), "00:00");
}

#[test]
fn test_md_segments_and_srt_agree_on_boundaries() {
    // The whole point of routing both through build_cues: the segment count
    // matches the SRT cue count for the same caps.
    let words: Vec<WordInfo> = (0..20)
        .map(|i| WordInfo {
            word: format!("word{i}"),
            start: i as f64,
            end: i as f64 + 0.4,
            confidence: 0.9,
            speaker: None,
        })
        .collect();
    let segments = to_segments(&words, 0, 5);
    let srt = to_srt(&words, 0, 5);
    let srt_cues = srt.matches("-->").count();
    assert_eq!(segments.len(), srt_cues);
}

// -----------------------------------------------------------------------
// Natural-boundary transcript segmenter (`to_transcript_segments`)
// -----------------------------------------------------------------------

#[test]
fn test_to_transcript_segments_empty() {
    let words: Vec<WordInfo> = Vec::new();
    let segments = to_transcript_segments(&words);
    assert!(segments.is_empty());
}

#[test]
fn test_to_transcript_segments_single_word() {
    let words = vec![WordInfo::new("привет", 0.0, 0.5, 0.98, None)];
    let segments = to_transcript_segments(&words);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, 0.0);
    assert_eq!(segments[0].end, 0.5);
    assert_eq!(segments[0].text, "привет");
    assert_eq!(segments[0].words.len(), 1);
    assert_eq!(segments[0].speaker, None);
}

#[test]
fn test_to_transcript_segments_split_on_pause() {
    // 1.1 s gap between the two words crosses the 0.9 s pause threshold.
    let words = vec![
        WordInfo::new("привет", 0.0, 0.5, 0.98, None),
        WordInfo::new("мир", 1.6, 2.0, 0.97, None),
    ];
    let segments = to_transcript_segments(&words);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].text, "привет");
    assert_eq!(segments[1].text, "мир");
}

#[test]
fn test_to_transcript_segments_split_on_punctuation() {
    // "привет." ends a sentence, so the next word starts a new segment.
    let words = vec![
        WordInfo::new("привет.", 0.0, 0.5, 0.98, None),
        WordInfo::new("мир", 0.6, 1.0, 0.97, None),
        WordInfo::new("как", 1.1, 1.5, 0.96, None),
    ];
    let segments = to_transcript_segments(&words);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].text, "привет.");
    assert_eq!(segments[1].text, "мир как");
}

#[test]
fn test_to_transcript_segments_split_on_speaker_change() {
    let words = vec![
        WordInfo::new("привет", 0.0, 0.5, 0.98, Some(0)),
        WordInfo::new("мир", 0.6, 1.0, 0.97, Some(1)),
    ];
    let segments = to_transcript_segments(&words);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].speaker, Some(0));
    assert_eq!(segments[1].speaker, Some(1));
}

#[test]
fn test_to_transcript_segments_split_on_max_duration() {
    // Generate 35 words with gaps just under the pause threshold so the
    // only reason to split is the 30 s duration cap.
    let words: Vec<WordInfo> = (0..35)
        .map(|i| {
            let start = i as f64 * 0.89;
            WordInfo::new(format!("word{i}"), start, start + 0.1, 0.95, None)
        })
        .collect();
    let segments = to_transcript_segments(&words);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start, 0.0);
    // The first segment ends where the 30 s cap is crossed.
    assert!(segments[0].end <= 30.0, "first segment exceeds cap");
    assert_eq!(
        segments[1].start,
        segments[0].words.last().unwrap().end + 0.79
    );
    // Every word is accounted for exactly once.
    let total: usize = segments.iter().map(|s| s.words.len()).sum();
    assert_eq!(total, 35);
}

#[test]
fn test_to_transcript_segments_speaker_omitted_when_none() {
    let words = vec![
        WordInfo::new("привет", 0.0, 0.5, 0.98, None),
        WordInfo::new("мир", 0.6, 1.0, 0.97, None),
    ];
    let segments = to_transcript_segments(&words);
    let json = serde_json::to_value(&segments).unwrap();
    assert!(json[0].get("speaker").is_none());
}

#[test]
fn test_to_transcript_segments_speaker_present_when_diarized() {
    let words = vec![
        WordInfo::new("привет", 0.0, 0.5, 0.98, Some(0)),
        WordInfo::new("мир", 0.6, 1.0, 0.97, Some(0)),
    ];
    let segments = to_transcript_segments(&words);
    let json = serde_json::to_value(&segments).unwrap();
    assert_eq!(json[0]["speaker"], 0);
}
