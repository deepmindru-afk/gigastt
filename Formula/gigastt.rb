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
  version "2.0.13"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.0.13/gigastt-2.0.13-aarch64-apple-darwin.tar.gz"
      sha256 "5178a7f96249bfec97f141cf7ab9f4727c95cded3a26d4e4f2360407e6812a5f"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.0.13/gigastt-2.0.13-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "a0934eccefdc00d5a7a3ddaca134c2ebb60785ba93a19900737471fc7085020a"
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
