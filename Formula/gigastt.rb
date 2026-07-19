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
  version "2.11.2"
  license "MIT"

  on_macos do
    # Apple Silicon only — GitHub retired the macos-13 Intel runners, so there is
    # no prebuilt x86_64-apple-darwin tarball. Intel Macs: `cargo install gigastt`.
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.11.2/gigastt-2.11.2-aarch64-apple-darwin.tar.gz"
      sha256 "8bc98cfd791d67f40129d7b29876769463e232806644603127e632eb76111aa5"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.11.2/gigastt-2.11.2-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "4a775b6a93add250686fdfd3a1dfa0df40c6c60d5e410031bc9493da252dacb7"
    elsif Hardware::CPU.arm?
      # sha256 is a placeholder; .github/workflows/homebrew.yml overwrites it
      # from SHA256SUMS.txt after the first release carrying this target.
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.11.2/gigastt-2.11.2-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "6c576270707854bf78aa97c4ced887e3c4126f12aaed355192354b840d7f7ce9"
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
