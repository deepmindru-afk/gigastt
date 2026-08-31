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
  version "2.19.0"
  license "MIT"

  on_macos do
    # Apple Silicon only — GitHub retired the macos-13 Intel runners, so there is
    # no prebuilt x86_64-apple-darwin tarball. Intel Macs: `cargo install gigastt`.
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.19.0/gigastt-2.19.0-aarch64-apple-darwin.tar.gz"
      sha256 "5077e0b2af120567b6f19a0e0245d382b13caa539933f4dd52fbe1ffe5b55794"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.19.0/gigastt-2.19.0-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "a1303f33c9198fbf9f4feeb69545bc521119b06ee0267e0943dfa8db23ffcfd7"
    elsif Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.19.0/gigastt-2.19.0-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "1f68f946576e861c3477fb1506f6cb3c916b1dc28c7f39850113e15a86584bf4"
    end
  end

  def install
    bin.install "gigastt"
  end

  def caveats
    <<~EOS
      The GigaAM v3 INT8 model (~225 MB) is downloaded on first run into
      ~/.gigastt/models (lean prequantized path; no FP32 step).

      Quick start:
        gigastt download         # lean INT8 bundle (~225 MB)
        gigastt serve            # starts STT server on 127.0.0.1:9876

      Homepage: https://github.com/ekhodzitsky/gigastt
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/gigastt --version")
  end
end
