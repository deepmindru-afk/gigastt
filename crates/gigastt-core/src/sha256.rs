//! Streaming SHA-256 and lowercase hex encoding for model content identity.
//!
//! RustCrypto selects CPU acceleration when available and otherwise uses its
//! portable implementation. Keep the NIST and streaming-boundary tests below.

use sha2::Digest;

/// Streaming SHA-256 hasher with a fixed-size digest and no allocation.
pub(crate) struct Sha256(sha2::Sha256);

impl Sha256 {
    pub(crate) fn new() -> Self {
        Self(sha2::Sha256::new())
    }

    pub(crate) fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    pub(crate) fn finalize(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 0x0f) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::{Sha256, hex_lower};

    fn digest(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        hex_lower(&h.finalize())
    }

    /// FIPS 180-4 / NIST CAVP vectors.
    #[test]
    fn test_sha256_matches_nist_vectors() {
        assert_eq!(
            digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// One million 'a' — exercises multi-block streaming and the length field.
    /// Multi-block hashing still runs under Miri via the chunking test; this
    /// vector does not finish inside the nightly budget.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "one million bytes does not finish under Miri; multi-block hashing is covered by the chunking test"
    )]
    fn test_sha256_million_a() {
        let mut h = Sha256::new();
        let chunk = vec![b'a'; 1000];
        for _ in 0..1000 {
            h.update(&chunk);
        }
        assert_eq!(
            hex_lower(&h.finalize()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    /// Feeding the same bytes in ragged pieces must not change the digest:
    /// the internal 64-byte block buffer has to stitch them correctly.
    #[test]
    fn test_sha256_chunking_is_transparent() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let mut whole = Sha256::new();
        whole.update(&data);
        let expected = hex_lower(&whole.finalize());

        for step in [1usize, 3, 7, 63, 64, 65, 127] {
            let mut h = Sha256::new();
            for piece in data.chunks(step) {
                h.update(piece);
            }
            assert_eq!(hex_lower(&h.finalize()), expected, "step {step}");
        }
    }

    #[test]
    fn test_hex_lower_pads_single_digit_bytes() {
        assert_eq!(hex_lower(&[0x00, 0x0f, 0xff]), "000fff");
    }
}
