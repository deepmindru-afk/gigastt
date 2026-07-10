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
  version "2.8.0"
  license "MIT"

  on_macos do
    # Apple Silicon only — GitHub retired the macos-13 Intel runners, so there is
    # no prebuilt x86_64-apple-darwin tarball. Intel Macs: `cargo install gigastt`.
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.8.0/gigastt-2.8.0-aarch64-apple-darwin.tar.gz"
      sha256 "fbffdd8611b29229ef080f5947571ba0a4ed99fe93e44636287f252f001d3318"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.8.0/gigastt-2.8.0-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "2e6d6f8bbfd2347d2cf2a93daec720cbcc88d10b0028ebf5abae819e8dc96154"
    elsif Hardware::CPU.arm?
      # sha256 is a placeholder; .github/workflows/homebrew.yml overwrites it
      # from SHA256SUMS.txt after the first release carrying this target.
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.8.0/gigastt-2.8.0-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "f646e9822f7e41f0633b36fe5c765f01798f9d383f5c318ac7591b980b3b4513"
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
