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
  version "2.0.10"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.0.10/gigastt-2.0.10-aarch64-apple-darwin.tar.gz"
      sha256 "bf7e107b0129ca0a08398c58f25e92abb3f98efed042b06a6931aebb25335a4c"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/gigastt/releases/download/v2.0.10/gigastt-2.0.10-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "c2b8073aa008130e4fbeb149adfcf2e42a68b9d2f17eda5585bc4077004f341a"
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
