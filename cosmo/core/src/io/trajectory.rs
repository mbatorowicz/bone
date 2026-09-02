//! Zapis trajektorii w kawałkach, z indeksem.
//!
//! Indeks trzyma numer kawałka i pozycję w kawałku dla każdej klatki, więc odczyt
//! klatki to jedno otwarcie pliku. Wersja przeglądająca wszystkie kawałki po kolei
//! kosztowała liniowo z długością nagrania, a odtwarzacz pyta o klatkę kilka razy na
//! sekundę.
//!
//! Paczka schodzi na dysk, gdy się zapełni ALBO gdy od poprzedniego zapisu minęło
//! `flush_interval`. Sam warunek zapełnienia nie wystarcza: przy paczce 64 klatek
//! i zapisie co 20 kroków pierwszy indeks pojawiałby się po 1280 krokach, a do tego
//! czasu odtwarzacz widzi zero klatek i wygląda na zepsuty.
//!
//! Klatki są zapisywane w `f32`. To dane do OGLĄDANIA, nie do wznawiania biegu —
//! wznawianie czyta checkpoint w `f64`. Przy 120 tys. cząstek i tysiącu klatek `f64`
//! oznaczałby 2,9 GB zamiast 1,4 GB, a różnicy nie da się zobaczyć na ekranie.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::io::binary::{
    expect_magic, read_f32_vec, read_f64, read_u32, write_f32_slice, write_f64, write_u32,
};
use crate::vec3::{vec3, Vec3};

const MAGIC: &[u8] = b"BONEFRM1";
pub const INDEX_FILE: &str = "trajectory.json";
const FRAMES_DIR: &str = "frames";

/// Gdzie leży dana klatka: numer kawałka i pozycja w nim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Location {
    pub chunk: u32,
    pub offset: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Index {
    pub n_frames: usize,
    pub stride: usize,
    pub frames: Vec<Location>,
}

/// Jedna klatka: położenia i odcień (β) każdego zapisanego punktu.
#[derive(Clone, Debug)]
pub struct Frame {
    pub time: f64,
    pub positions: Vec<Vec3>,
    pub shades: Vec<f32>,
}

pub struct Writer {
    out: PathBuf,
    chunk_size: usize,
    stride: usize,
    flush_interval: Duration,
    last_flush: Instant,
    pending: Vec<Frame>,
    chunk: u32,
    index: Vec<Location>,
}

impl Writer {
    pub fn new(out_dir: impl AsRef<Path>, stride: usize) -> io::Result<Self> {
        let out = out_dir.as_ref().to_path_buf();
        fs::create_dir_all(out.join(FRAMES_DIR))?;
        Ok(Self {
            out,
            chunk_size: 64,
            stride: stride.max(1),
            flush_interval: Duration::from_secs(2),
            last_flush: Instant::now(),
            pending: Vec::new(),
            chunk: 0,
            index: Vec::new(),
        })
    }

    pub fn n_frames(&self) -> usize {
        self.index.len()
    }

    /// Dopisz klatkę; zapisuje na dysk, gdy paczka dojrzeje.
    ///
    /// Odcień jest podawany jako funkcja indeksu, a nie gotowa tablica, bo znaczy coś
    /// innego w każdym modelu (`β` w trybie SR, kontrast gęstości w ΛCDM) i liczy się
    /// tylko dla cząstek, które faktycznie trafiają do zapisu. Ten moduł nie ma powodu
    /// wiedzieć, co to za wielkość.
    pub fn push(
        &mut self,
        time: f64,
        all_positions: &[Vec3],
        shade: impl Fn(usize) -> f32,
    ) -> io::Result<()> {
        let positions: Vec<Vec3> = all_positions.iter().step_by(self.stride).copied().collect();
        let shades: Vec<f32> = (0..all_positions.len())
            .step_by(self.stride)
            .map(shade)
            .collect();
        self.index.push(Location {
            chunk: self.chunk,
            offset: self.pending.len() as u32,
        });
        self.pending.push(Frame {
            time,
            positions,
            shades,
        });
        let stale = self.last_flush.elapsed() >= self.flush_interval;
        if self.pending.len() >= self.chunk_size || stale {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return self.write_index();
        }
        self.last_flush = Instant::now();
        let path = self.out.join(FRAMES_DIR).join(chunk_name(self.chunk));
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(MAGIC)?;
        write_u32(&mut out, self.pending.len() as u32)?;
        write_u32(&mut out, self.pending[0].positions.len() as u32)?;
        for frame in &self.pending {
            write_f64(&mut out, frame.time)?;
            let mut flat = Vec::with_capacity(frame.positions.len() * 3);
            for p in &frame.positions {
                flat.extend_from_slice(&[p.x as f32, p.y as f32, p.z as f32]);
            }
            write_f32_slice(&mut out, &flat)?;
            write_f32_slice(&mut out, &frame.shades)?;
        }
        drop(out);

        self.chunk += 1;
        self.pending.clear();
        self.write_index()
    }

    fn write_index(&self) -> io::Result<()> {
        let index = Index {
            n_frames: self.index.len(),
            stride: self.stride,
            frames: self.index.clone(),
        };
        let text = serde_json::to_string(&index)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        fs::write(self.out.join(INDEX_FILE), text)
    }
}

impl Drop for Writer {
    /// Domknięcie nagrania nawet wtedy, gdy bieg został przerwany.
    ///
    /// Bez tego klatki z ostatniej, niezapełnionej paczki przepadałyby po każdym
    /// przerwaniu — czyli dokładnie w sytuacji, w której najbardziej chce się
    /// zobaczyć, co się stało na końcu.
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

fn chunk_name(chunk: u32) -> String {
    format!("chunk_{chunk:05}.bin")
}

pub fn read_index(out_dir: impl AsRef<Path>) -> Index {
    fs::read_to_string(out_dir.as_ref().join(INDEX_FILE))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Wczytaj jedną klatkę po numerze.
pub fn load_frame(out_dir: impl AsRef<Path>, frame: usize) -> Option<Frame> {
    let dir = out_dir.as_ref();
    let index = read_index(dir);
    let location = *index.frames.get(frame)?;
    let path = dir.join(FRAMES_DIR).join(chunk_name(location.chunk));
    let mut input = BufReader::new(File::open(path).ok()?);
    expect_magic(&mut input, MAGIC).ok()?;
    let count = read_u32(&mut input).ok()?;
    let points = read_u32(&mut input).ok()? as usize;
    if location.offset >= count {
        return None;
    }
    for _ in 0..location.offset {
        read_f64(&mut input).ok()?;
        read_f32_vec(&mut input, points * 3).ok()?;
        read_f32_vec(&mut input, points).ok()?;
    }
    let time = read_f64(&mut input).ok()?;
    let flat = read_f32_vec(&mut input, points * 3).ok()?;
    let shades = read_f32_vec(&mut input, points).ok()?;
    Some(Frame {
        time,
        positions: flat
            .chunks_exact(3)
            .map(|c| vec3(c[0] as f64, c[1] as f64, c[2] as f64))
            .collect(),
        shades,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::vec3;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bone-traj-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn cloud(n: usize) -> Vec<Vec3> {
        (0..n)
            .map(|i| vec3(i as f64, -(i as f64) * 0.5, i as f64 * 0.25))
            .collect()
    }

    /// Odcień zależny od indeksu, żeby pomieszanie kolejności było widoczne.
    fn shade(i: usize) -> f32 {
        (i % 100) as f32 / 100.0
    }

    #[test]
    fn frames_round_trip() {
        let dir = temp_dir("roundtrip");
        let points = cloud(50);
        let mut writer = Writer::new(&dir, 1).unwrap();
        for k in 0..5 {
            writer.push(k as f64 * 0.5, &points, shade).unwrap();
        }
        writer.flush().unwrap();
        assert_eq!(writer.n_frames(), 5);
        drop(writer);

        for k in 0..5 {
            let frame = load_frame(&dir, k).expect("klatka istnieje");
            assert!((frame.time - k as f64 * 0.5).abs() < 1e-12);
            assert_eq!(frame.positions.len(), 50);
            assert_eq!(frame.shades.len(), 50);
            assert!((frame.positions[7].x - 7.0).abs() < 1e-5);
            assert!((frame.shades[7] - shade(7)).abs() < 1e-6);
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stride_thins_the_points_and_keeps_them_paired() {
        let dir = temp_dir("stride");
        let mut writer = Writer::new(&dir, 4).unwrap();
        writer.push(0.0, &cloud(100), shade).unwrap();
        writer.flush().unwrap();
        drop(writer);

        let frame = load_frame(&dir, 0).unwrap();
        assert_eq!(frame.positions.len(), 25);
        assert_eq!(read_index(&dir).stride, 4);
        // Piąty zapisany punkt to cząstka nr 20 — położenie i odcień muszą pochodzić
        // z tej samej cząstki, inaczej mapa kolorów jest przypisana losowo.
        assert!((frame.positions[5].x - 20.0).abs() < 1e-5);
        assert!((frame.shades[5] - shade(20)).abs() < 1e-6);
        fs::remove_dir_all(&dir).ok();
    }

    /// Nagranie dłuższe niż jedna paczka musi być czytelne po numerze klatki —
    /// to jest cały powód istnienia indeksu.
    #[test]
    fn index_spans_multiple_chunks() {
        let dir = temp_dir("chunks");
        let points = cloud(10);
        let mut writer = Writer::new(&dir, 1).unwrap();
        for k in 0..150 {
            writer.push(k as f64, &points, shade).unwrap();
        }
        writer.flush().unwrap();
        drop(writer);

        let index = read_index(&dir);
        assert_eq!(index.n_frames, 150);
        let chunks: std::collections::BTreeSet<u32> = index.frames.iter().map(|l| l.chunk).collect();
        assert!(chunks.len() > 1, "wszystko wylądowało w jednej paczce");

        for k in [0usize, 63, 64, 100, 149] {
            let frame = load_frame(&dir, k).unwrap_or_else(|| panic!("brak klatki {k}"));
            assert!((frame.time - k as f64).abs() < 1e-12, "klatka {k}");
        }
        fs::remove_dir_all(&dir).ok();
    }

    /// Przerwany bieg nie może zgubić ostatniej paczki.
    #[test]
    fn dropping_the_writer_flushes_pending_frames() {
        let dir = temp_dir("drop");
        let points = cloud(20);
        {
            let mut writer = Writer::new(&dir, 1).unwrap();
            writer.push(0.0, &points, shade).unwrap();
            writer.push(1.0, &points, shade).unwrap();
        }
        assert_eq!(read_index(&dir).n_frames, 2);
        assert!(load_frame(&dir, 1).is_some());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_recording_reads_as_empty() {
        let dir = temp_dir("missing");
        let index = read_index(&dir);
        assert_eq!(index.n_frames, 0);
        assert!(load_frame(&dir, 0).is_none());
    }

    #[test]
    fn out_of_range_frame_is_none() {
        let dir = temp_dir("range");
        let mut writer = Writer::new(&dir, 1).unwrap();
        writer.push(0.0, &cloud(10), shade).unwrap();
        writer.flush().unwrap();
        drop(writer);
        assert!(load_frame(&dir, 99).is_none());
        fs::remove_dir_all(&dir).ok();
    }

    /// Zerowy krok próbkowania jest podniesiony do jedynki: dzielenie przez zero
    /// w `step_by` panikuje, a suwak w panelu potrafi zejść do zera.
    #[test]
    fn zero_stride_is_treated_as_one() {
        let dir = temp_dir("zero-stride");
        let mut writer = Writer::new(&dir, 0).unwrap();
        writer.push(0.0, &cloud(8), shade).unwrap();
        writer.flush().unwrap();
        drop(writer);
        assert_eq!(load_frame(&dir, 0).unwrap().positions.len(), 8);
        fs::remove_dir_all(&dir).ok();
    }
}
