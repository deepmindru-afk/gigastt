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
  version "2.14.3"
  license "MIT"

  on_macos do
    # Apple Silicon only — GitHub retired the macos-13 Intel runners, so there is
    # no prebuilt x86_64-apple-darwin tarball. Intel Macs: `cargo install gigastt`.
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.14.3/gigastt-2.14.3-aarch64-apple-darwin.tar.gz"
      sha256 "b4e7354ae276d8675b7c4d1d3fcee484628a2e812050fade50a24aff97db1f62"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.14.3/gigastt-2.14.3-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "b93727f205b8204ea8473f70577a8976439a5408bdfb7b8018f5192d5b7cfd90"
    elsif Hardware::CPU.arm?
      # sha256 is a placeholder; .github/workflows/homebrew.yml overwrites it
      # from SHA256SUMS.txt after the first release carrying this target.
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.14.3/gigastt-2.14.3-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "bd08d8d7f856e107e2e60ca1c9d0a7fadef79e3c866f5175c5ce4e9f9476637b"
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
