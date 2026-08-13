use super::*;

#[test]
fn test_export_params_parse_codec_query() {
    // The query string is how REST clients pass the pair; pin it through
    // axum's Query extractor itself (the exact extraction path the
    // handler uses).
    let uri: axum::http::Uri = "http://localhost/v1/transcribe?codec=pcmu&sample_rate=8000"
        .parse()
        .unwrap();
    let axum::extract::Query(params) =
        axum::extract::Query::<ExportParams>::try_from_uri(&uri).expect("codec query must parse");
    assert_eq!(params.codec.as_deref(), Some("pcmu"));
    assert_eq!(params.sample_rate, Some(8000));
}

#[tokio::test]
async fn test_render_export_default_returns_none() {
    let result = sample_export_result();
    let params = ExportParams::default();
    assert!(render_export_response(&result, &params).unwrap().is_none());
}

#[tokio::test]
async fn test_render_export_txt() {
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("txt".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    assert_eq!(body, "привет мир");
}

#[tokio::test]
async fn test_render_export_srt_content_type() {
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("srt".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/x-subrip; charset=utf-8"
    );
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("[SPEAKER_0] привет мир"));
}

#[tokio::test]
async fn test_render_export_vtt_download_header() {
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("vtt".into()),
        download: Some("recording.vtt".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(
        resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
        "attachment; filename=\"recording.vtt\"; filename*=UTF-8''recording.vtt"
    );
}

#[tokio::test]
async fn test_render_export_download_filename_with_control_char_does_not_panic() {
    // The download filename is user-controlled; control characters must not
    // produce an invalid header value / panic — they are sanitized out of
    // the quoted fallback and percent-encoded in `filename*`.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("srt".into()),
        download: Some("evil\r\nInjected: x".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
        "attachment; filename=\"evil__Injected: x\"; filename*=UTF-8''evil%0D%0AInjected%3A%20x"
    );
}

#[tokio::test]
async fn test_render_export_invalid_format() {
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("docx".into()),
        ..ExportParams::default()
    };
    let err = render_export_response(&result, &params).unwrap_err();
    assert_eq!(err.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_render_export_invalid_format_body_code() {
    // The invalid-format error carries the machine-readable `invalid_format`
    // code so clients can distinguish it from other 400s.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("xml".into()),
        ..ExportParams::default()
    };
    let err = render_export_response(&result, &params).unwrap_err();
    assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(err.into_body(), 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["code"], "invalid_format");
}

#[tokio::test]
async fn test_render_export_uppercase_json_returns_none() {
    // Format negotiation is case-insensitive: an explicit (any-case) "json"
    // means "keep the default TranscribeResponse contract", so the helper
    // returns None instead of building a Response.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("JSON".into()),
        ..ExportParams::default()
    };
    assert!(render_export_response(&result, &params).unwrap().is_none());
}

#[tokio::test]
async fn test_render_export_uppercase_format_renders() {
    // Non-JSON format strings are also case-insensitive (parsed via
    // ExportFormat::from_str), so "SRT" still renders subtitles.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("SRT".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/x-subrip; charset=utf-8"
    );
}

#[tokio::test]
async fn test_render_export_empty_download_uses_default_name() {
    // An empty `download` value still requests an attachment; the helper
    // synthesizes the default `transcript.<ext>` filename.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("vtt".into()),
        download: Some(String::new()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(
        resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
        "attachment; filename=\"transcript.vtt\"; filename*=UTF-8''transcript.vtt"
    );
}

#[tokio::test]
async fn test_render_export_download_filename_injection_neutralized() {
    // A crafted `download` value trying to splice a second `filename*`
    // parameter must survive only as inert data: the quote becomes `_` in
    // the fallback, and the `filename*` bytes are percent-encoded, so the
    // spoofed `spoofed.exe` never appears as a real header parameter.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("srt".into()),
        download: Some("evil\"; filename*=UTF-8''spoofed.exe".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(
        resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
        "attachment; filename=\"evil_; filename*=UTF-8''spoofed.exe\"; \
         filename*=UTF-8''evil%22%3B%20filename%2A%3DUTF-8%27%27spoofed.exe"
    );
}

#[tokio::test]
async fn test_render_export_download_filename_unicode_percent_encoded() {
    // Non-ASCII names get an ASCII-safe fallback for legacy clients and
    // keep the full UTF-8 name percent-encoded in `filename*` (RFC 6266).
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("txt".into()),
        download: Some("é.txt".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(
        resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
        "attachment; filename=\"_.txt\"; filename*=UTF-8''%C3%A9.txt"
    );
}

#[tokio::test]
async fn test_render_export_download_filename_cyrillic_percent_encoded() {
    // Cyrillic input must not leak raw non-ASCII bytes into the header:
    // the fallback replaces each non-ASCII character, `filename*` carries
    // the percent-encoded UTF-8, and the whole value stays ASCII.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("srt".into()),
        download: Some("отчёт.srt".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    let value = resp
        .headers()
        .get(header::CONTENT_DISPOSITION)
        .unwrap()
        .to_str()
        .unwrap();
    let encoded_name = format!("{}.srt", "%d0%be%d1%82%d1%87%d1%91%d1%82".to_uppercase());
    assert_eq!(
        value,
        format!("attachment; filename=\"_____.srt\"; filename*=UTF-8''{encoded_name}")
    );
    assert!(value.is_ascii());
}

#[tokio::test]
async fn test_render_export_md_includes_word_timestamps() {
    // The Markdown path honours `word_timestamps` and renders the per-word
    // table; the content type is text/markdown.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("md".into()),
        word_timestamps: Some(true),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/markdown; charset=utf-8"
    );
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("# Transcript"));
    assert!(text.contains("| Word | Start | End |"));
}

#[tokio::test]
async fn test_render_export_line_break_opts_passed_through() {
    // Tight per-line caps must be threaded into RenderOpts so the rendered
    // subtitles actually break — proving the params override the defaults.
    let result = sample_export_result();
    let loose = ExportParams {
        format: Some("srt".into()),
        ..ExportParams::default()
    };
    let tight = ExportParams {
        format: Some("srt".into()),
        max_words_per_line: Some(1),
        ..ExportParams::default()
    };
    let loose_resp = render_export_response(&result, &loose).unwrap().unwrap();
    let tight_resp = render_export_response(&result, &tight).unwrap().unwrap();
    let loose_body = axum::body::to_bytes(loose_resp.into_body(), 4096)
        .await
        .unwrap();
    let tight_body = axum::body::to_bytes(tight_resp.into_body(), 4096)
        .await
        .unwrap();
    let loose_text = String::from_utf8(loose_body.to_vec()).unwrap();
    let tight_text = String::from_utf8(tight_body.to_vec()).unwrap();
    // One word per line yields one cue per word (more "-->" arrows) than the
    // default 14-words-per-line grouping.
    let loose_cues = loose_text.matches("-->").count();
    let tight_cues = tight_text.matches("-->").count();
    assert!(
        tight_cues > loose_cues,
        "tight={tight_cues} should exceed loose={loose_cues}"
    );
}

#[tokio::test]
async fn test_render_export_md_segments_emits_headers() {
    // `format=md` + `segments=true` switches Markdown to `### [mm:ss]`
    // section headers over the cue boundaries, dropping the flat
    // `# Transcript` blob.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("md".into()),
        segments: Some(true),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/markdown; charset=utf-8"
    );
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("### [00:00]"));
    assert!(text.contains("[SPEAKER_0] привет мир"));
    // Segment mode replaces the flat transcript blob.
    assert!(!text.contains("# Transcript"));
}

#[tokio::test]
async fn test_render_export_md_without_segments_unchanged() {
    // Plain `format=md` (no segments) keeps the existing frontmatter +
    // `# Transcript` layout — segment mode is strictly opt-in.
    let result = sample_export_result();
    let params = ExportParams {
        format: Some("md".into()),
        ..ExportParams::default()
    };
    let resp = render_export_response(&result, &params).unwrap().unwrap();
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("# Transcript"));
    assert!(!text.contains("### ["));
}

#[tokio::test]
async fn test_render_export_segments_ignored_for_srt() {
    // `segments=true` is a JSON/Markdown affordance; SRT is already
    // cue-based and must render identically with or without the flag.
    let result = sample_export_result();
    let plain = ExportParams {
        format: Some("srt".into()),
        ..ExportParams::default()
    };
    let with_segments = ExportParams {
        format: Some("srt".into()),
        segments: Some(true),
        ..ExportParams::default()
    };
    let a = render_export_response(&result, &plain).unwrap().unwrap();
    let b = render_export_response(&result, &with_segments)
        .unwrap()
        .unwrap();
    let a_body = axum::body::to_bytes(a.into_body(), 4096).await.unwrap();
    let b_body = axum::body::to_bytes(b.into_body(), 4096).await.unwrap();
    assert_eq!(a_body, b_body);
}

#[test]
fn test_export_params_deserialize_from_query() {
    // The query-param shape drives format negotiation; confirm axum's Query
    // extractor maps every field so the handler sees the caller's choices.
    let uri: axum::http::Uri = "http://x/?format=srt&download=out.srt&max_chars_per_line=20&max_words_per_line=3&word_timestamps=true&segments=true&channels=split&diarization=true"
        .parse()
        .unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert_eq!(params.format.as_deref(), Some("srt"));
    assert_eq!(params.download.as_deref(), Some("out.srt"));
    assert_eq!(params.max_chars_per_line, Some(20));
    assert_eq!(params.max_words_per_line, Some(3));
    assert_eq!(params.word_timestamps, Some(true));
    assert_eq!(params.segments, Some(true));
    assert_eq!(params.channels.as_deref(), Some("split"));
    assert_eq!(params.diarization, Some(true));
}

#[test]
fn test_export_params_default_empty_query() {
    // No query params -> all None, which the handler maps to JSON defaults.
    let uri: axum::http::Uri = "http://x/".parse().unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert!(params.format.is_none());
    assert!(params.download.is_none());
    assert!(params.max_chars_per_line.is_none());
    // The per-request knob overrides default to absent so the handler falls
    // back to the engine's boot policy (byte-unchanged response).
    assert!(params.punctuation.is_none());
    assert!(params.itn.is_none());
    assert!(params.vad.is_none());
    assert!(params.hotwords.is_none());
    assert!(params.hotwords_boost.is_none());
    assert!(params.variant.is_none());
}

#[test]
fn test_transcribe_knob_params_deserialize_from_query() {
    // `?punctuation=false&itn=false&vad=false&variant=rnnt` maps to
    // `Some(false)`/`Some("rnnt")`, letting the handler override the boot
    // policy per request.
    let uri: axum::http::Uri = "http://x/?punctuation=false&itn=false&vad=false&variant=rnnt"
        .parse()
        .unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert_eq!(params.punctuation, Some(false));
    assert_eq!(params.itn, Some(false));
    assert_eq!(params.vad, Some(false));
    assert_eq!(params.variant.as_deref(), Some("rnnt"));

    // The `true` direction deserializes symmetrically.
    let uri: axum::http::Uri = "http://x/?punctuation=true&itn=true&vad=true"
        .parse()
        .unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert_eq!(params.punctuation, Some(true));
    assert_eq!(params.itn, Some(true));
    assert_eq!(params.vad, Some(true));
}

#[test]
fn test_hotwords_query_param_parsing() {
    // Comma-separated phrases + optional boost deserialize from the query
    // string and map to HotwordOverride via hotwords_from_export_params.
    let uri: axum::http::Uri =
        "http://x/?hotwords=%D1%81%D0%B1%D0%B5%D1%80,%D1%82%D0%B8%D0%BD%D1%8C%D0%BA%D0%BE%D1%84%D1%84&hotwords_boost=3.5"
            .parse()
            .unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert_eq!(params.hotwords.as_deref(), Some("сбер,тинькофф"));
    assert_eq!(params.hotwords_boost, Some(3.5));

    let hw = hotwords_from_export_params(&params).expect("hotwords present");
    assert_eq!(hw.phrases, vec!["сбер".to_string(), "тинькофф".to_string()]);
    assert_eq!(hw.boost, Some(3.5));

    // Absent hotwords → engine default (None), even if boost is set alone.
    let uri: axum::http::Uri = "http://x/?hotwords_boost=9".parse().unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    assert!(hotwords_from_export_params(&params).is_none());

    // Empty key present → Some(empty) force-off.
    let uri: axum::http::Uri = "http://x/?hotwords=".parse().unwrap();
    let Query(params): Query<ExportParams> = Query::try_from_uri(&uri).unwrap();
    let hw = hotwords_from_export_params(&params).expect("key present means override");
    assert!(hw.phrases.is_empty());

    assert_eq!(
        parse_hotwords_query(" сбер , , тинькофф "),
        vec!["сбер".to_string(), "тинькофф".to_string()]
    );
    assert!(parse_hotwords_query("").is_empty());
    assert!(parse_hotwords_query(",,,").is_empty());
}
