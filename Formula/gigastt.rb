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
  version "2.20.0"
  license "MIT"

  on_macos do
    # Apple Silicon only — GitHub retired the macos-13 Intel runners, so there is
    # no prebuilt x86_64-apple-darwin tarball. Intel Macs: `cargo install gigastt`.
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.20.0/gigastt-2.20.0-aarch64-apple-darwin.tar.gz"
      sha256 "cc081c512ecf536c72ca8f7276048dbd7eee1979383e1d554b27b9f39e0eca08"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.20.0/gigastt-2.20.0-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "4d7e9fffc837cdb22ca70ede32b1635c6022c40fcc4e1ed616666d5cfe1ff536"
    elsif Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.20.0/gigastt-2.20.0-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "0af1f06c56ec20ad14ad9cf481a1938c7526881c78d14db8e38d0edc0790e489"
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
