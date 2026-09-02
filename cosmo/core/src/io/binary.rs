//! Odczyt i zapis liczb w ustalonej kolejności bajtów.
//!
//! Wszystko jest little-endian, jawnie, a nie „jak wyjdzie na tej maszynie".
//! Checkpoint zapisany na jednym komputerze musi się otworzyć na drugim, a
//! `to_ne_bytes` daje plik poprawny wyłącznie tam, gdzie powstał — i to bez żadnego
//! objawu poza bezsensownymi liczbami.

use std::io::{self, Read, Write};

pub fn write_u32(out: &mut impl Write, value: u32) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

pub fn write_u64(out: &mut impl Write, value: u64) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

pub fn write_f64(out: &mut impl Write, value: f64) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

pub fn write_f64_slice(out: &mut impl Write, values: &[f64]) -> io::Result<()> {
    let mut buffer = Vec::with_capacity(values.len() * 8);
    for v in values {
        buffer.extend_from_slice(&v.to_le_bytes());
    }
    out.write_all(&buffer)
}

pub fn write_f32_slice(out: &mut impl Write, values: &[f32]) -> io::Result<()> {
    let mut buffer = Vec::with_capacity(values.len() * 4);
    for v in values {
        buffer.extend_from_slice(&v.to_le_bytes());
    }
    out.write_all(&buffer)
}

pub fn read_u32(input: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    input.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub fn read_u64(input: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    input.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

pub fn read_f64(input: &mut impl Read) -> io::Result<f64> {
    let mut b = [0u8; 8];
    input.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

pub fn read_f64_vec(input: &mut impl Read, count: usize) -> io::Result<Vec<f64>> {
    let mut bytes = vec![0u8; count * 8];
    input.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("kawałek ma 8 bajtów")))
        .collect())
}

pub fn read_f32_vec(input: &mut impl Read, count: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; count * 4];
    input.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("kawałek ma 4 bajty")))
        .collect())
}

/// Sprawdź sygnaturę pliku.
///
/// # Errors
/// Gdy sygnatura się nie zgadza — wtedy plik jest z innego formatu albo z innej
/// wersji, a czytanie go dalej dałoby losowe liczby zamiast błędu.
pub fn expect_magic(input: &mut impl Read, magic: &[u8]) -> io::Result<()> {
    let mut got = vec![0u8; magic.len()];
    input.read_exact(&mut got)?;
    if got != magic {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "obcy format pliku: oczekiwano {:?}, jest {:?}",
                String::from_utf8_lossy(magic),
                String::from_utf8_lossy(&got)
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_round_trip() {
        let mut buffer = Vec::new();
        write_u32(&mut buffer, 0xDEAD_BEEF).unwrap();
        write_u64(&mut buffer, u64::MAX - 7).unwrap();
        write_f64(&mut buffer, -3.5e120).unwrap();
        let mut cursor = buffer.as_slice();
        assert_eq!(read_u32(&mut cursor).unwrap(), 0xDEAD_BEEF);
        assert_eq!(read_u64(&mut cursor).unwrap(), u64::MAX - 7);
        assert_eq!(read_f64(&mut cursor).unwrap(), -3.5e120);
    }

    #[test]
    fn slices_round_trip() {
        let f64s = vec![1.0, -2.5, 1e-300, f64::MAX];
        let f32s = vec![0.5f32, -1.25, 3.75];
        let mut buffer = Vec::new();
        write_f64_slice(&mut buffer, &f64s).unwrap();
        write_f32_slice(&mut buffer, &f32s).unwrap();
        let mut cursor = buffer.as_slice();
        assert_eq!(read_f64_vec(&mut cursor, f64s.len()).unwrap(), f64s);
        assert_eq!(read_f32_vec(&mut cursor, f32s.len()).unwrap(), f32s);
    }

    #[test]
    fn byte_order_is_little_endian_regardless_of_host() {
        let mut buffer = Vec::new();
        write_u32(&mut buffer, 1).unwrap();
        assert_eq!(buffer, vec![1, 0, 0, 0]);
    }

    #[test]
    fn foreign_magic_is_rejected() {
        let mut cursor: &[u8] = b"NOPE1234";
        let err = expect_magic(&mut cursor, b"BONECKP1").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn matching_magic_is_accepted_and_consumed() {
        let mut cursor: &[u8] = b"BONECKP1rest";
        expect_magic(&mut cursor, b"BONECKP1").unwrap();
        assert_eq!(cursor, b"rest");
    }
}
