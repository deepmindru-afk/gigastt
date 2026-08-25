use super::*;
use gigastt_core::inference::WordInfo;

fn sample_result() -> TranscribeResult {
    TranscribeResult {
        text: "привет мир".into(),
        words: vec![
            WordInfo::new("привет", 0.0, 0.5, 0.98, None),
            WordInfo::new("мир", 1.6, 2.0, 0.97, None),
        ],
        duration_s: 2.0,
        confidence: Some(0.975),
    }
}

#[test]
fn test_response_format_parse() {
    assert_eq!(
        OpenAIResponseFormat::parse("json").unwrap(),
        OpenAIResponseFormat::Json
    );
    assert_eq!(
        OpenAIResponseFormat::parse("TEXT").unwrap(),
        OpenAIResponseFormat::Text
    );
    assert_eq!(
        OpenAIResponseFormat::parse("verbose_json").unwrap(),
        OpenAIResponseFormat::VerboseJson
    );
    assert!(OpenAIResponseFormat::parse("docx").is_err());
}

#[test]
fn test_normalize_language() {
    assert_eq!(normalize_language("Russian"), "ru");
    assert_eq!(normalize_language("en-US"), "en");
    assert_eq!(normalize_language(""), "ru");
    assert_eq!(normalize_language("kk"), "kk");
    assert_eq!(normalize_language("pt-BR"), "pt-br");
}

#[test]
fn test_apply_form_fields_and_finalize() {
    let mut opts = OpenAITranscriptionOptions::default();
    apply_openai_form_field(&mut opts, "model", b"whisper-1").unwrap();
    apply_openai_form_field(&mut opts, "language", b"Russian").unwrap();
    apply_openai_form_field(&mut opts, "response_format", b"verbose_json").unwrap();
    apply_openai_form_field(&mut opts, "timestamp_granularities[]", b"word").unwrap();
    apply_openai_form_field(&mut opts, "prompt", b"ignored").unwrap();
    apply_openai_form_field(&mut opts, "temperature", b"0").unwrap();
    finalize_openai_options(&mut opts).unwrap();

    assert_eq!(opts.model.as_deref(), Some("whisper-1"));
    assert_eq!(opts.language.as_deref(), Some("Russian"));
    assert_eq!(opts.response_format, OpenAIResponseFormat::VerboseJson);
    assert!(opts.include_words);
    // word-only request must not force segments on.
    assert!(!opts.include_segments);
}

#[test]
fn test_finalize_defaults_segments_for_verbose() {
    let mut opts = OpenAITranscriptionOptions {
        response_format: OpenAIResponseFormat::VerboseJson,
        ..Default::default()
    };
    finalize_openai_options(&mut opts).unwrap();
    assert!(opts.include_segments);
    assert!(!opts.include_words);
}

#[test]
fn test_stream_flag_and_incompatible_format() {
    let mut opts = OpenAITranscriptionOptions::default();
    apply_openai_form_field(&mut opts, "stream", b"true").unwrap();
    assert!(opts.stream);
    finalize_openai_options(&mut opts).unwrap();

    let mut opts = OpenAITranscriptionOptions {
        stream: true,
        response_format: OpenAIResponseFormat::Srt,
        ..Default::default()
    };
    let err = finalize_openai_options(&mut opts).unwrap_err();
    assert!(err.contains("stream=true"));
}

#[test]
fn test_stream_assembler_grows_and_finalizes() {
    let mut a = OpenAIStreamAssembler::new();
    assert_eq!(a.push_segment("привет", false).as_deref(), Some("привет"));
    assert_eq!(a.push_segment("привет мир", false).as_deref(), Some(" мир"));
    assert_eq!(a.push_segment("привет мир", true).as_deref(), None);
    assert_eq!(a.text(), "привет мир");
    // Second utterance
    assert_eq!(
        a.push_segment("как дела", true).as_deref(),
        Some(" как дела")
    );
    assert_eq!(a.text(), "привет мир как дела");
}

#[test]
fn test_sse_payloads() {
    let d = sse_delta_payload(" hi");
    let v: serde_json::Value = serde_json::from_str(&d).unwrap();
    assert_eq!(v["type"], "transcript.text.delta");
    assert_eq!(v["delta"], " hi");
    let done = sse_done_payload("hello");
    let v: serde_json::Value = serde_json::from_str(&done).unwrap();
    assert_eq!(v["type"], "transcript.text.done");
    assert_eq!(v["text"], "hello");
}

#[test]
fn test_invalid_response_format_field() {
    let mut opts = OpenAITranscriptionOptions::default();
    let err = apply_openai_form_field(&mut opts, "response_format", b"pdf").unwrap_err();
    assert!(err.contains("response_format"));
}

#[test]
fn test_json_response_is_text_only() {
    let resp = OpenAIJsonResponse {
        text: "привет".into(),
    };
    let v = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["text"], "привет");
    assert_eq!(v.as_object().unwrap().len(), 1);
}

#[test]
fn test_verbose_default_has_segments_no_words() {
    let result = sample_result();
    let mut opts = OpenAITranscriptionOptions {
        response_format: OpenAIResponseFormat::VerboseJson,
        language: Some("ru".into()),
        ..Default::default()
    };
    finalize_openai_options(&mut opts).unwrap();
    let v = serde_json::to_value(build_verbose_response(&result, &opts)).unwrap();
    assert_eq!(v["task"], "transcribe");
    assert_eq!(v["language"], "ru");
    assert_eq!(v["duration"], 2.0);
    assert_eq!(v["text"], "привет мир");
    assert!(v["segments"].is_array());
    // Two words with a long pause → two segments.
    assert_eq!(v["segments"].as_array().unwrap().len(), 2);
    assert!(v.get("words").is_none());
    // Whisper-shaped segment fields present.
    assert_eq!(v["segments"][0]["id"], 0);
    assert!(
        v["segments"][0]["text"]
            .as_str()
            .unwrap()
            .contains("привет")
    );
}

#[test]
fn test_verbose_word_granularity() {
    let result = sample_result();
    let mut opts = OpenAITranscriptionOptions {
        response_format: OpenAIResponseFormat::VerboseJson,
        include_words: true,
        include_segments: false,
        language: Some("English".into()),
        ..Default::default()
    };
    finalize_openai_options(&mut opts).unwrap();
    let v = serde_json::to_value(build_verbose_response(&result, &opts)).unwrap();
    assert_eq!(v["language"], "en");
    assert!(v.get("segments").is_none());
    assert_eq!(v["words"].as_array().unwrap().len(), 2);
    assert_eq!(v["words"][0]["word"], "привет");
    assert_eq!(v["words"][0]["start"], 0.0);
    assert_eq!(v["words"][0]["end"], 0.5);
}

#[test]
fn test_verbose_both_granularities() {
    let result = sample_result();
    let mut opts = OpenAITranscriptionOptions {
        response_format: OpenAIResponseFormat::VerboseJson,
        include_words: true,
        include_segments: true,
        ..Default::default()
    };
    finalize_openai_options(&mut opts).unwrap();
    let v = serde_json::to_value(build_verbose_response(&result, &opts)).unwrap();
    assert!(v["segments"].is_array());
    assert!(v["words"].is_array());
}

#[test]
fn test_render_json_and_text_content_types() {
    let result = sample_result();
    let json_opts = OpenAITranscriptionOptions::default();
    let resp = render_openai_response(&result, &json_opts);
    assert_eq!(resp.status(), StatusCode::OK);

    let text_opts = OpenAITranscriptionOptions {
        response_format: OpenAIResponseFormat::Text,
        ..Default::default()
    };
    let resp = render_openai_response(&result, &text_opts);
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.starts_with("text/plain"), "got {ct}");
}

#[test]
fn test_render_srt_and_vtt() {
    let result = sample_result();
    for fmt in [OpenAIResponseFormat::Srt, OpenAIResponseFormat::Vtt] {
        let opts = OpenAITranscriptionOptions {
            response_format: fmt,
            ..Default::default()
        };
        let resp = render_openai_response(&result, &opts);
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

/// Multipart helper shared by integration-style unit tests.
fn multipart_body(boundary: &str, fields: &[(&str, Option<&str>, &[u8])]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, filename, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        match filename {
            Some(fname) => body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\n\
                     Content-Type: application/octet-stream\r\n\r\n"
                )
                .as_bytes(),
            ),
            None => body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            ),
        }
        body.extend_from_slice(value);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

#[tokio::test]
async fn test_parse_multipart_full_form() {
    use axum::Router;
    use axum::routing::post;

    let app = Router::new().route(
        "/t",
        post(|multipart: Multipart| async move {
            match parse_openai_multipart(multipart).await {
                Ok(req) => {
                    let v = serde_json::json!({
                        "file_len": req.file.len(),
                        "model": req.options.model,
                        "language": req.options.language,
                        "format": match req.options.response_format {
                            OpenAIResponseFormat::VerboseJson => "verbose_json",
                            OpenAIResponseFormat::Json => "json",
                            OpenAIResponseFormat::Text => "text",
                            OpenAIResponseFormat::Srt => "srt",
                            OpenAIResponseFormat::Vtt => "vtt",
                        },
                        "words": req.options.include_words,
                        "segments": req.options.include_segments,
                    });
                    (StatusCode::OK, Json(v)).into_response()
                }
                Err(resp) => *resp,
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let boundary = "----gigasttTestBoundary";
    let file_bytes = b"RIFF....fake-wav-payload";
    let body = multipart_body(
        boundary,
        &[
            ("model", None, b"whisper-1"),
            ("language", None, b"ru"),
            ("response_format", None, b"verbose_json"),
            ("timestamp_granularities[]", None, b"word"),
            ("timestamp_granularities[]", None, b"segment"),
            ("file", Some("clip.wav"), file_bytes),
        ],
    );

    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/t"))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(v["file_len"], file_bytes.len());
    assert_eq!(v["model"], "whisper-1");
    assert_eq!(v["language"], "ru");
    assert_eq!(v["format"], "verbose_json");
    assert_eq!(v["words"], true);
    assert_eq!(v["segments"], true);
}

#[tokio::test]
async fn test_parse_multipart_missing_file() {
    use axum::Router;
    use axum::routing::post;

    let app = Router::new().route(
        "/t",
        post(|multipart: Multipart| async move {
            match parse_openai_multipart(multipart).await {
                Ok(_) => StatusCode::OK.into_response(),
                Err(resp) => *resp,
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let boundary = "----gigasttTestBoundary";
    let body = multipart_body(boundary, &[("model", None, b"whisper-1")]);
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/t"))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let v: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(v["code"], "missing_file");
}

#[tokio::test]
async fn test_parse_multipart_invalid_format() {
    use axum::Router;
    use axum::routing::post;

    let app = Router::new().route(
        "/t",
        post(|multipart: Multipart| async move {
            match parse_openai_multipart(multipart).await {
                Ok(_) => StatusCode::OK.into_response(),
                Err(resp) => *resp,
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let boundary = "----gigasttTestBoundary";
    let body = multipart_body(
        boundary,
        &[
            ("response_format", None, b"pdf"),
            ("file", Some("a.wav"), b"x"),
        ],
    );
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/t"))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let v: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(v["code"], "invalid_response_format");
}

#[tokio::test]
async fn test_parse_multipart_empty_file() {
    use axum::Router;
    use axum::routing::post;

    let app = Router::new().route(
        "/t",
        post(|multipart: Multipart| async move {
            match parse_openai_multipart(multipart).await {
                Ok(_) => StatusCode::OK.into_response(),
                Err(resp) => *resp,
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let boundary = "----gigasttTestBoundary";
    let body = multipart_body(boundary, &[("file", Some("empty.wav"), b"")]);
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/t"))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let v: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
    assert_eq!(v["code"], "empty_body");
}
