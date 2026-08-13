use super::*;

#[test]
fn test_parse_model_variant_valid_and_invalid() {
    assert_eq!(parse_model_variant("rnnt").unwrap(), ModelVariant::Rnnt);
    assert_eq!(
        parse_model_variant("e2e_rnnt").unwrap(),
        ModelVariant::E2eRnnt
    );
    assert!(parse_model_variant("whisper").is_err());
}

#[test]
fn test_parse_progress_mode_value_parser() {
    assert_eq!(parse_progress_mode("human").unwrap(), ProgressMode::Human);
    assert_eq!(parse_progress_mode("json").unwrap(), ProgressMode::Json);
    assert!(parse_progress_mode("xml").is_err());
}

#[test]
fn test_cli_rejects_unknown_subcommand() {
    let res = Cli::try_parse_from(["gigastt", "bogus"]);
    assert!(res.is_err(), "unknown subcommand must be rejected");
}

#[test]
fn test_cli_top_level_long_help_points_to_subcommand_engine_flags() {
    use clap::CommandFactory;
    let help = Cli::command().render_long_help().to_string();
    for needle in [
        "serve --help",
        "--punctuation",
        "--itn",
        "--vad",
        "--model-variant",
    ] {
        assert!(
            help.contains(needle),
            "top-level long help must mention `{needle}`:\n{help}"
        );
    }
}
