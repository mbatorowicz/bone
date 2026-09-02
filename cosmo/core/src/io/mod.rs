//! Trwały zapis biegu: checkpoint do wznawiania i trajektoria do oglądania.
//!
//! Dwa formaty, bo dwa różne zadania. Checkpoint musi być dokładny co do bitu,
//! więc trzyma `f64` i cały stan. Trajektoria musi być mała, bo klatek jest tysiące,
//! więc trzyma `f32` i tylko to, co widać na ekranie: położenia i odcień.

pub mod binary;
pub mod checkpoint;
pub mod trajectory;
