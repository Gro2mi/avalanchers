//! Minimal blosc1 chunk decoder.
//!
//! Zarr stores written by this project use the blosc codec with the zstd
//! compressor and the bitshuffle filter. Decoding is implemented here in pure
//! Rust (via `ruzstd`) so that it also works on `wasm32-unknown-unknown`, where
//! the C based `zstd`/`blosc` bindings cannot be built.

use std::io::Read;

const HEADER_LEN: usize = 16;

const FLAG_SHUFFLE: u8 = 0x01;
const FLAG_MEMCPYED: u8 = 0x02;
const FLAG_BITSHUFFLE: u8 = 0x04;

const COMPRESSOR_ZSTD: u8 = 4;

fn compressor_name(code: u8) -> &'static str {
    match code {
        0 => "blosclz",
        1 => "lz4",
        2 => "snappy",
        3 => "zlib",
        4 => "zstd",
        _ => "unknown",
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BloscError {
    #[error("Truncated blosc chunk: expected at least {expected} bytes, found {found}")]
    Truncated { expected: usize, found: usize },

    #[error("Unsupported blosc compressor \"{}\"; only zstd is supported", compressor_name(*.0))]
    UnsupportedCompressor(u8),

    #[error("Failed to decompress blosc block: {0}")]
    Decompress(String),

    #[error("Blosc block decoded to {found} bytes, expected {expected}")]
    BlockSizeMismatch { expected: usize, found: usize },
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, BloscError> {
    let end = offset + 4;
    if bytes.len() < end {
        return Err(BloscError::Truncated {
            expected: end,
            found: bytes.len(),
        });
    }
    Ok(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

fn decompress_zstd(input: &[u8], expected: usize) -> Result<Vec<u8>, BloscError> {
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(input)
        .map_err(|e| BloscError::Decompress(e.to_string()))?;
    let mut out = Vec::with_capacity(expected);
    decoder
        .read_to_end(&mut out)
        .map_err(|e| BloscError::Decompress(e.to_string()))?;
    Ok(out)
}

/// Decodes a standalone zstd frame, e.g. a Zarr chunk using the `zstd` codec.
pub fn decode_zstd_frame(input: &[u8]) -> Result<Vec<u8>, BloscError> {
    decompress_zstd(input, input.len() * 4)
}

/// Decodes a blosc1 chunk into its raw bytes.
pub fn decode_blosc(chunk: &[u8]) -> Result<Vec<u8>, BloscError> {
    if chunk.len() < HEADER_LEN {
        return Err(BloscError::Truncated {
            expected: HEADER_LEN,
            found: chunk.len(),
        });
    }

    let flags = chunk[2];
    let typesize = chunk[3] as usize;
    let nbytes = read_u32(chunk, 4)? as usize;
    let blocksize = read_u32(chunk, 8)? as usize;

    if flags & FLAG_MEMCPYED != 0 {
        let end = HEADER_LEN + nbytes;
        if chunk.len() < end {
            return Err(BloscError::Truncated {
                expected: end,
                found: chunk.len(),
            });
        }
        return Ok(chunk[HEADER_LEN..end].to_vec());
    }

    let compressor = (flags >> 5) & 0x07;
    if compressor != COMPRESSOR_ZSTD {
        return Err(BloscError::UnsupportedCompressor(compressor));
    }

    if blocksize == 0 {
        return Ok(Vec::new());
    }
    let nblocks = nbytes.div_ceil(blocksize);

    let mut out = Vec::with_capacity(nbytes);
    for block_index in 0..nblocks {
        let start = read_u32(chunk, HEADER_LEN + block_index * 4)? as usize;
        let block_size = blocksize.min(nbytes - block_index * blocksize);

        let compressed_len = read_u32(chunk, start)? as usize;
        let data_start = start + 4;
        let data_end = data_start + compressed_len;
        if chunk.len() < data_end {
            return Err(BloscError::Truncated {
                expected: data_end,
                found: chunk.len(),
            });
        }
        let payload = &chunk[data_start..data_end];

        // blosc stores a block verbatim when compression did not pay off.
        let block = if compressed_len == block_size {
            payload.to_vec()
        } else {
            decompress_zstd(payload, block_size)?
        };

        if block.len() != block_size {
            return Err(BloscError::BlockSizeMismatch {
                expected: block_size,
                found: block.len(),
            });
        }

        // The filters are applied per block, using that block's size.
        let block = if flags & FLAG_BITSHUFFLE != 0 {
            bitunshuffle(&block, typesize)
        } else if flags & FLAG_SHUFFLE != 0 {
            unshuffle(&block, typesize)
        } else {
            block
        };

        out.extend_from_slice(&block);
    }

    Ok(out)
}

/// Reverses blosc's byte-shuffle filter.
fn unshuffle(block: &[u8], typesize: usize) -> Vec<u8> {
    if typesize <= 1 {
        return block.to_vec();
    }
    let items = block.len() / typesize;
    let mut out = vec![0u8; block.len()];
    for byte_index in 0..typesize {
        let offset = byte_index * items;
        for item in 0..items {
            out[item * typesize + byte_index] = block[offset + item];
        }
    }
    // Trailing bytes that do not fill a whole item are stored verbatim.
    let tail = items * typesize;
    out[tail..].copy_from_slice(&block[tail..]);
    out
}

/// Reverses the bitshuffle filter, mirroring `bshuf_untrans_bit_elem`.
fn bitunshuffle(block: &[u8], typesize: usize) -> Vec<u8> {
    if typesize == 0 {
        return block.to_vec();
    }
    let size = block.len() / typesize;
    // blosc leaves blocks whose element count is not a multiple of 8 untouched.
    if size == 0 || !size.is_multiple_of(8) {
        return block.to_vec();
    }

    let mut tmp = vec![0u8; block.len()];
    untrans_bitrow_eight(block, &mut tmp, size, typesize);
    let mut out = vec![0u8; block.len()];
    shuffle_bit_eightelem(&tmp, &mut out, size, typesize);
    out
}

fn untrans_bitrow_eight(input: &[u8], out: &mut [u8], size: usize, typesize: usize) {
    let nbyte_bitrow = size / 8;
    let row_bytes = 8 * typesize;
    for hh in 0..nbyte_bitrow {
        for row in 0..row_bytes {
            out[hh * row_bytes + row] = input[row * nbyte_bitrow + hh];
        }
    }
}

/// Transposes each 8x8 bit matrix, mirroring `bshuf_shuffle_bit_eightelem`.
fn shuffle_bit_eightelem(input: &[u8], out: &mut [u8], size: usize, typesize: usize) {
    let nbyte = typesize * size;
    let row_bytes = 8 * typesize;

    for jj in (0..row_bytes).step_by(8) {
        let mut ii = 0;
        while ii + row_bytes <= nbyte {
            let offset = ii + jj;
            let x = u64::from_le_bytes(input[offset..offset + 8].try_into().unwrap());
            let x = transpose_bit_8x8(x);
            let bytes = x.to_le_bytes();
            for (kk, byte) in bytes.iter().enumerate() {
                out[ii + jj / 8 + kk * typesize] = *byte;
            }
            ii += row_bytes;
        }
    }
}

/// Transposes an 8x8 bit matrix packed into a u64.
fn transpose_bit_8x8(mut x: u64) -> u64 {
    let mut t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x = x ^ t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x = x ^ t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x = x ^ t ^ (t << 28);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes with the same codec the writer uses, then decodes it back.
    #[cfg(not(all(target_os = "unknown", target_arch = "wasm32")))]
    fn roundtrip(
        values: &[f32],
        shuffle: zarrs::array::codec::bytes_to_bytes::blosc::BloscShuffleMode,
    ) {
        use zarrs::array::BytesToBytesCodecTraits;
        use zarrs::array::codec::bytes_to_bytes::blosc::{
            BloscCodec, BloscCompressionLevel, BloscCompressor,
        };

        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let codec = BloscCodec::new(
            BloscCompressor::Zstd,
            BloscCompressionLevel::try_from(1u8).unwrap(),
            None, // automatic blocksize
            shuffle,
            Some(std::mem::size_of::<f32>()),
        )
        .unwrap();

        let encoded = codec
            .encode(raw.clone().into(), &zarrs::array::CodecOptions::default())
            .unwrap();

        let decoded = decode_blosc(&encoded).expect("decode failed");
        assert_eq!(decoded, raw, "roundtrip mismatch for {shuffle:?}");
    }

    #[cfg(not(all(target_os = "unknown", target_arch = "wasm32")))]
    #[test]
    fn test_decode_blosc_bitshuffle_roundtrip() {
        use zarrs::array::codec::bytes_to_bytes::blosc::BloscShuffleMode;
        let values: Vec<f32> = (0..4096).map(|i| i as f32 * 0.5 - 100.0).collect();
        roundtrip(&values, BloscShuffleMode::BitShuffle);
    }

    #[cfg(not(all(target_os = "unknown", target_arch = "wasm32")))]
    #[test]
    fn test_decode_blosc_byteshuffle_roundtrip() {
        use zarrs::array::codec::bytes_to_bytes::blosc::BloscShuffleMode;
        let values: Vec<f32> = (0..4096).map(|i| i as f32 * 0.5 - 100.0).collect();
        roundtrip(&values, BloscShuffleMode::Shuffle);
    }

    #[cfg(not(all(target_os = "unknown", target_arch = "wasm32")))]
    #[test]
    fn test_decode_blosc_noshuffle_roundtrip() {
        use zarrs::array::codec::bytes_to_bytes::blosc::BloscShuffleMode;
        let values: Vec<f32> = (0..4096).map(|i| i as f32 * 0.5 - 100.0).collect();
        roundtrip(&values, BloscShuffleMode::NoShuffle);
    }

    /// Incompressible data exercises the verbatim block and multi-block paths.
    #[cfg(not(all(target_os = "unknown", target_arch = "wasm32")))]
    #[test]
    fn test_decode_blosc_incompressible_roundtrip() {
        use zarrs::array::codec::bytes_to_bytes::blosc::BloscShuffleMode;
        let mut seed = 0x12345678u32;
        let values: Vec<f32> = (0..8192)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                f32::from_bits(seed)
            })
            .filter(|v| v.is_finite())
            .collect();
        roundtrip(&values, BloscShuffleMode::BitShuffle);
    }

    /// A DEM-sized array spans several blosc blocks, like the stores on disk.
    #[cfg(not(all(target_os = "unknown", target_arch = "wasm32")))]
    #[test]
    fn test_decode_blosc_bitshuffle_multiple_blocks() {
        use zarrs::array::codec::bytes_to_bytes::blosc::BloscShuffleMode;
        let (width, height) = (554usize, 489usize);
        let values: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f32, (i / width) as f32);
                1200.0 + 300.0 * (x / 40.0).sin() + 150.0 * (y / 55.0).cos()
            })
            .collect();
        roundtrip(&values, BloscShuffleMode::BitShuffle);
    }

    #[test]
    fn test_decode_blosc_rejects_truncated_chunk() {
        let result = decode_blosc(&[0u8; 4]);
        assert!(matches!(result, Err(BloscError::Truncated { .. })));
    }

    #[test]
    fn test_transpose_bit_8x8_is_an_involution() {
        for seed in [0u64, 1, 0x0123_4567_89AB_CDEF, u64::MAX] {
            assert_eq!(transpose_bit_8x8(transpose_bit_8x8(seed)), seed);
        }
    }
}
