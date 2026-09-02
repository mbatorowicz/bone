//! Odtwarzanie zapisanej trajektorii — te same klatki, bez ponownego liczenia.

use std::path::{Path, PathBuf};

use bone_core::grid::center_span;
use bone_core::io::trajectory::{self, Frame, Index};
use bone_core::vec3::Vec3;

pub struct Replay {
    dir: PathBuf,
    index: Index,
    frame: Frame,
    cursor: usize,
}

impl Replay {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref().to_path_buf();
        let index = trajectory::read_index(&dir);
        if index.n_frames == 0 || index.frames.is_empty() {
            return Err(format!("brak klatek w {}", dir.display()));
        }
        let frame = trajectory::load_frame(&dir, 0)
            .ok_or_else(|| format!("nie czytam klatki 0 z {}", dir.display()))?;
        Ok(Self {
            dir,
            index,
            frame,
            cursor: 0,
        })
    }

    pub fn n_frames(&self) -> usize {
        self.index.n_frames.max(self.index.frames.len())
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn seek(&mut self, frame: usize) {
        let last = self.n_frames().saturating_sub(1);
        let frame = frame.min(last);
        if frame == self.cursor {
            return;
        }
        if let Some(loaded) = trajectory::load_frame(&self.dir, frame) {
            self.frame = loaded;
            self.cursor = frame;
        }
    }

    pub fn headline(&self) -> String {
        format!(
            "odtwarzanie  klatka {}/{}  t={:.4}  N={}",
            self.cursor + 1,
            self.n_frames(),
            self.frame.time,
            self.frame.positions.len()
        )
    }

    pub fn rows(&self) -> Vec<(&'static str, String)> {
        vec![
            ("klatka", format!("{:>10}", self.cursor + 1)),
            ("z / t", format!("{:>10.4}", self.frame.time)),
            ("N", format!("{:>10}", self.frame.positions.len())),
        ]
    }

    pub fn n(&self) -> usize {
        self.frame.positions.len()
    }

    pub fn position(&self, index: usize) -> Vec3 {
        self.frame.positions[index]
    }

    pub fn shade(&self, index: usize) -> f32 {
        self.frame.shades.get(index).copied().unwrap_or(0.0).clamp(0.0, 1.0)
    }

    pub fn center_span(&self) -> (Vec3, f64) {
        center_span(&self.frame.positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bone_core::session::Session;
    use bone_core::sr;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bone-replay-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn small_sr() -> sr::Config {
        let mut cfg = sr::presets::galaxy();
        cfg.spawn.n_particles = 64;
        cfg.solver.backend = sr::config::BackendKind::Exact;
        cfg.run.trajectory_every = 1;
        cfg
    }

    #[test]
    fn replay_reads_recorded_frames() {
        let dir = temp_dir("open");
        let mut session = Session::start_sr(small_sr(), dir.clone(), true).unwrap();
        session.advance(3).unwrap();
        session.flush_recording().unwrap();
        let replay = Replay::open(&dir).expect("odtwarzacz czyta nagranie");
        assert!(replay.n_frames() >= 1);
        assert_eq!(replay.n(), 64);
        std::fs::remove_dir_all(&dir).ok();
    }
}
