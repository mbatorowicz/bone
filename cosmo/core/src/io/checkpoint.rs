//! Zapis i odczyt pełnego stanu.
//!
//! Zapisujemy PĘD, nie prędkość. Prędkość zależy od `c`, więc checkpoint zapisany
//! z prędkościami zmieniałby fizykę po wczytaniu z innym `c` — pęd jest niezależną
//! od tego zmienną stanu.
//!
//! Stan idzie do pliku binarnego, a konfiguracja obok, do JSON-a. Rozdzielenie jest
//! celowe: tablice mają megabajty i nikt ich nie czyta okiem, a konfigurację czyta
//! się i edytuje ręcznie stale. Wsadzenie jej do tego samego pliku binarnego
//! oznaczałoby pisanie narzędzia do podejrzenia dwunastu liczb.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::io::binary::{
    expect_magic, read_f64, read_f64_vec, read_u64, write_f64, write_f64_slice, write_u64,
};
use crate::lcdm::{self, Cosmology};
use crate::sr::config::Config;
use crate::sr::state::State;
use crate::vec3::{vec3, Vec3};

const MAGIC_SR: &[u8] = b"BONECKP1";
const MAGIC_LCDM: &[u8] = b"BONELCD1";
pub const STATE_FILE: &str = "checkpoint.bin";
pub const CONFIG_FILE: &str = "config.json";

/// Który model leży w katalogu — rozstrzyga sygnatura, nie nazwa pliku.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Relativistic,
    Cosmological,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct LcdmFile {
    model: String,
    cosmology: Cosmology,
    run: lcdm::RunConfig,
}

pub fn save(state: &State, cfg: &Config, out_dir: impl AsRef<Path>) -> io::Result<PathBuf> {
    let dir = out_dir.as_ref();
    fs::create_dir_all(dir)?;

    let path = dir.join(STATE_FILE);
    let mut out = BufWriter::new(File::create(&path)?);
    out.write_all(MAGIC_SR)?;
    write_u64(&mut out, state.n() as u64)?;
    write_f64(&mut out, state.time)?;
    write_u64(&mut out, state.step)?;
    write_f64_slice(&mut out, &flatten(&state.positions))?;
    write_f64_slice(&mut out, &flatten(&state.momenta))?;
    write_f64_slice(&mut out, &state.masses)?;
    drop(out);

    fs::write(dir.join(CONFIG_FILE), cfg.to_json())?;
    Ok(path)
}

/// Sama konfiguracja zapisanego biegu, bez wczytywania stanu.
///
/// Rozdzielone od [`load`], bo wznawianie musi znać konfigurację ZANIM zapadnie
/// decyzja o starcie, a tablice czyta się później. Ładowanie kilku megabajtów tylko
/// po to, żeby zajrzeć w parametry, byłoby marnotrawstwem.
pub fn load_config(out_dir: impl AsRef<Path>) -> Option<Config> {
    let text = fs::read_to_string(out_dir.as_ref().join(CONFIG_FILE)).ok()?;
    Config::from_json(&text).ok()
}

pub fn load(out_dir: impl AsRef<Path>) -> io::Result<(State, Config)> {
    let dir = out_dir.as_ref();
    let mut input = BufReader::new(File::open(dir.join(STATE_FILE))?);
    expect_magic(&mut input, MAGIC_SR)?;
    let n = read_u64(&mut input)? as usize;
    let time = read_f64(&mut input)?;
    let step = read_u64(&mut input)?;
    let positions = unflatten(read_f64_vec(&mut input, n * 3)?);
    let momenta = unflatten(read_f64_vec(&mut input, n * 3)?);
    let masses = read_f64_vec(&mut input, n)?;

    let mut state = State::new(positions, momenta, masses)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    state.time = time;
    state.step = step;
    Ok((state, load_config(dir).unwrap_or_default()))
}

pub fn exists(out_dir: impl AsRef<Path>) -> bool {
    out_dir.as_ref().join(STATE_FILE).is_file()
}

pub fn kind(out_dir: impl AsRef<Path>) -> Option<Kind> {
    let mut magic = [0u8; 8];
    let mut file = File::open(out_dir.as_ref().join(STATE_FILE)).ok()?;
    use std::io::Read;
    file.read_exact(&mut magic).ok()?;
    match &magic {
        m if m == MAGIC_SR => Some(Kind::Relativistic),
        m if m == MAGIC_LCDM => Some(Kind::Cosmological),
        _ => None,
    }
}

pub fn save_lcdm(engine: &lcdm::Engine, out_dir: impl AsRef<Path>) -> io::Result<PathBuf> {
    let dir = out_dir.as_ref();
    fs::create_dir_all(dir)?;

    let path = dir.join(STATE_FILE);
    let mut out = BufWriter::new(File::create(&path)?);
    out.write_all(MAGIC_LCDM)?;
    write_u64(&mut out, engine.n() as u64)?;
    write_f64(&mut out, engine.a)?;
    write_u64(&mut out, engine.step)?;
    write_f64(&mut out, engine.mass)?;
    write_f64(&mut out, engine.box_size)?;
    write_f64(&mut out, engine.initial_contrast)?;
    write_f64_slice(&mut out, &flatten(&engine.positions))?;
    write_f64_slice(&mut out, &flatten(&engine.momenta))?;
    drop(out);

    let text = serde_json::to_string_pretty(&LcdmFile {
        model: "lcdm".to_string(),
        cosmology: engine.cosmology,
        run: engine.cfg,
    })
    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(dir.join(CONFIG_FILE), text)?;
    Ok(path)
}

pub fn load_lcdm(out_dir: impl AsRef<Path>) -> io::Result<lcdm::Engine> {
    let dir = out_dir.as_ref();
    let mut input = BufReader::new(File::open(dir.join(STATE_FILE))?);
    expect_magic(&mut input, MAGIC_LCDM)?;
    let n = read_u64(&mut input)? as usize;
    let a = read_f64(&mut input)?;
    let step = read_u64(&mut input)?;
    let mass = read_f64(&mut input)?;
    let box_size = read_f64(&mut input)?;
    let initial_contrast = read_f64(&mut input)?;
    let positions = unflatten(read_f64_vec(&mut input, n * 3)?);
    let momenta = unflatten(read_f64_vec(&mut input, n * 3)?);

    let text = fs::read_to_string(dir.join(CONFIG_FILE))?;
    let file: LcdmFile = serde_json::from_str(&text)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if file.model != "lcdm" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("config.json ma model „{}”, a nie lcdm", file.model),
        ));
    }
    Ok(lcdm::Engine::with_state(lcdm::Saved {
        cosmology: file.cosmology,
        cfg: file.run,
        a,
        positions,
        momenta,
        mass,
        box_size,
        step,
        initial_contrast,
    }))
}

fn flatten(v: &[Vec3]) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len() * 3);
    for p in v {
        out.extend_from_slice(&[p.x, p.y, p.z]);
    }
    out
}

fn unflatten(flat: Vec<f64>) -> Vec<Vec3> {
    flat.chunks_exact(3).map(|c| vec3(c[0], c[1], c[2])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sr::presets;
    use crate::sr::spawn;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bone-test-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn sample() -> (State, Config) {
        let mut cfg = presets::precision();
        cfg.spawn.n_particles = 64;
        let mut state = spawn::make_state(&cfg).state;
        state.time = 1.25;
        state.step = 42;
        (state, cfg)
    }

    #[test]
    fn state_survives_round_trip_bit_for_bit() {
        let dir = temp_dir("ckp-roundtrip");
        let (state, cfg) = sample();
        save(&state, &cfg, &dir).unwrap();
        let (back, _) = load(&dir).unwrap();

        assert_eq!(back.n(), state.n());
        assert_eq!(back.time, state.time);
        assert_eq!(back.step, state.step);
        // Bit w bit: f64 zapisany i odczytany nie ma prawa się zmienić, a
        // zaokrąglenie przy zapisie zmieniałoby fizykę po wznowieniu.
        for i in 0..state.n() {
            assert_eq!(back.positions[i], state.positions[i], "cząstka {i}");
            assert_eq!(back.momenta[i], state.momenta[i], "cząstka {i}");
            assert_eq!(back.masses[i], state.masses[i], "cząstka {i}");
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_survives_round_trip() {
        let dir = temp_dir("ckp-config");
        let (state, cfg) = sample();
        save(&state, &cfg, &dir).unwrap();
        assert_eq!(load(&dir).unwrap().1, cfg);
        assert_eq!(load_config(&dir), Some(cfg));
        fs::remove_dir_all(&dir).ok();
    }

    /// Wczytany stan nie ma pola sił — musi zostać przeliczone, bo zależy od G i ε,
    /// które mogły się zmienić między zapisem a wznowieniem.
    #[test]
    fn loaded_state_has_no_stale_field() {
        let dir = temp_dir("ckp-field");
        let (mut state, cfg) = sample();
        state.forces = Some(vec![crate::vec3::ZERO; state.n()]);
        save(&state, &cfg, &dir).unwrap();
        let (back, _) = load(&dir).unwrap();
        assert!(back.forces.is_none());
        assert!(back.potential.is_none());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn exists_reports_presence() {
        let dir = temp_dir("ckp-exists");
        assert!(!exists(&dir));
        let (state, cfg) = sample();
        save(&state, &cfg, &dir).unwrap();
        assert!(exists(&dir));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_config_falls_back_to_defaults() {
        let dir = temp_dir("ckp-noconfig");
        let (state, cfg) = sample();
        save(&state, &cfg, &dir).unwrap();
        fs::remove_file(dir.join(CONFIG_FILE)).unwrap();
        assert_eq!(load(&dir).unwrap().1, Config::default());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn foreign_file_is_rejected_not_misread() {
        let dir = temp_dir("ckp-foreign");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(STATE_FILE), b"to nie jest checkpoint").unwrap();
        assert!(load(&dir).is_err());
        fs::remove_dir_all(&dir).ok();
    }

    fn small_lcdm() -> lcdm::Engine {
        lcdm::Engine::new(
            lcdm::Cosmology::planck18(),
            lcdm::RunConfig {
                n_grid: 8,
                pm_grid: 16,
                ..lcdm::RunConfig::linear_growth()
            },
        )
    }

    #[test]
    fn lcdm_state_survives_round_trip() {
        let dir = temp_dir("ckp-lcdm");
        let mut engine = small_lcdm();
        engine.advance();
        engine.advance();
        let a = engine.a;
        let step = engine.step;
        let first = engine.positions[0];
        save_lcdm(&engine, &dir).unwrap();
        assert_eq!(kind(&dir), Some(Kind::Cosmological));
        let back = load_lcdm(&dir).unwrap();
        assert_eq!(back.step, step);
        assert_eq!(back.a, a);
        assert_eq!(back.positions[0], first);
        assert_eq!(back.n(), engine.n());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn kind_distinguishes_the_two_models() {
        let sr_dir = temp_dir("ckp-kind-sr");
        let (state, cfg) = sample();
        save(&state, &cfg, &sr_dir).unwrap();
        assert_eq!(kind(&sr_dir), Some(Kind::Relativistic));
        fs::remove_dir_all(&sr_dir).ok();
    }
}
