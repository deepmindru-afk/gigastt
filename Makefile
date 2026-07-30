.PHONY: check fix

check:
	cargo fmt --all --check
	cargo clippy --workspace -- -D warnings -A dead_code
	# --lib --bins: a bare `cargo test --workspace` pulls in the ~2.5h WER
	# benchmark, which is a `harness = false` target and so is not skipped by
	# `--ignored`. Matches .githooks/pre-commit.
	cargo test --workspace --lib --bins

fix:
	cargo fmt --all
	cargo clippy --workspace --fix -- -D warnings -A dead_code
