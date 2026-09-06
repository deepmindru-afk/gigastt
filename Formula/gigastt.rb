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
  version "2.21.0"
  license "MIT"

  on_macos do
    # Apple Silicon only — GitHub retired the macos-13 Intel runners, so there is
    # no prebuilt x86_64-apple-darwin tarball. Intel Macs: `cargo install gigastt`.
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.21.0/gigastt-2.21.0-aarch64-apple-darwin.tar.gz"
      sha256 "9a6c53997b3abf370b35debd68aa7ed085556ff6ec865d63d1eeb3cbae85accf"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.21.0/gigastt-2.21.0-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "b660fbdbde40054c59a8ac4740cd447daa6123a4e334c09f1a76deecb0e89aa7"
    elsif Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.21.0/gigastt-2.21.0-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "f4e45f9b13ea30fa5e9de7d24ab398d736eefff55c5d7ae09f306065750e50c5"
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
