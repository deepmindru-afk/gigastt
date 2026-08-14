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
  version "2.18.0"
  license "MIT"

  on_macos do
    # Apple Silicon only — GitHub retired the macos-13 Intel runners, so there is
    # no prebuilt x86_64-apple-darwin tarball. Intel Macs: `cargo install gigastt`.
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.18.0/gigastt-2.18.0-aarch64-apple-darwin.tar.gz"
      sha256 "4057d14c134071391f86c905f95e0366ff8fbc5e9520d723d92a76caea798379"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.18.0/gigastt-2.18.0-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "7814ea335f8aefb935ae75ba21c105ea81394c48e3c5ee31f9e7dd502a70dfee"
    elsif Hardware::CPU.arm?
      # sha256 is a placeholder; .github/workflows/homebrew.yml overwrites it
      # from SHA256SUMS.txt after the first release carrying this target.
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.18.0/gigastt-2.18.0-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "9c971dbb8bf54d8e552525402b1de47b4bc4fca710c3e9e53dfb298aba2d0f05"
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
