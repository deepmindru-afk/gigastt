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
  version "2.13.0"
  license "MIT"

  on_macos do
    # Apple Silicon only — GitHub retired the macos-13 Intel runners, so there is
    # no prebuilt x86_64-apple-darwin tarball. Intel Macs: `cargo install gigastt`.
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.13.0/gigastt-2.13.0-aarch64-apple-darwin.tar.gz"
      sha256 "97d6dd43940c8a75a620fb84fc51af45b71fbf0bc8a2030696ee91ffbd054a39"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.13.0/gigastt-2.13.0-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "d3ba6ee5d9db5431dce37aabc9fcc3a65c429970b59c4b01a92a22a9e4e794a1"
    elsif Hardware::CPU.arm?
      # sha256 is a placeholder; .github/workflows/homebrew.yml overwrites it
      # from SHA256SUMS.txt after the first release carrying this target.
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.13.0/gigastt-2.13.0-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "39a7234e17d039200db762e675b432a475401d6a48f9ebbdea3b90fddbe87187"
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
