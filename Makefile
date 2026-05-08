.PHONY: check fix

check:
	cargo fmt --all --check
	cargo clippy --workspace -- -D warnings -A dead_code
	cargo test --workspace

fix:
	cargo fmt --all
	cargo clippy --workspace --fix -- -D warnings -A dead_code
