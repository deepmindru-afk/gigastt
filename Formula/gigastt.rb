# Homebrew formula for gigastt.
#
# Install with:
#   brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
#   brew install gigastt
#
# The `sha256` values below are pinned to the v<version> release tarballs.
# They are refreshed automatically by the `.github/workflows/homebrew.yml`
# workflow after every successful `release.yml` run — do not hand-edit
# unless you are backfilling a release that predated that automation.

class Gigastt < Formula
  desc "On-device Russian speech recognition server powered by GigaAM v3"
  homepage "https://github.com/ekhodzitsky/gigastt"
  version "2.14.2"
  license "MIT"

  on_macos do
    # Apple Silicon only — GitHub retired the macos-13 Intel runners, so there is
    # no prebuilt x86_64-apple-darwin tarball. Intel Macs: `cargo install gigastt`.
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.14.2/gigastt-2.14.2-aarch64-apple-darwin.tar.gz"
      sha256 "b58c73f535e188a5abf6a8870f15d4ccb40ba4879ad0051756c025750447ff2e"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.14.2/gigastt-2.14.2-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "68c1a5899ee206ad7d33f80014acf7de6251467941c3a4844b27a36b96d7a488"
    elsif Hardware::CPU.arm?
      # sha256 is a placeholder; .github/workflows/homebrew.yml overwrites it
      # from SHA256SUMS.txt after the first release carrying this target.
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.14.2/gigastt-2.14.2-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "1756a8fc1571169f0bac7129a896fcf5396a219d2c5e62e58ca5f865d99019f5"
    end
  end

  def install
    bin.install "gigastt"
  end

  def caveats
    <<~EOS
      The GigaAM v3 model (~850 MB) is downloaded on first run into
      ~/.gigastt/models. An INT8-quantized encoder is produced automatically
      (~2 min one-time). Disable with `--skip-quantize` or
      `GIGASTT_SKIP_QUANTIZE=1`.

      Quick start:
        gigastt download         # fetches model + quantizes
        gigastt serve            # starts STT server on 127.0.0.1:9876

      Homepage: https://github.com/ekhodzitsky/gigastt
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/gigastt --version")
  end
end
