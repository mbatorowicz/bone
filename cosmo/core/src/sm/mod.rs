//! Laboratorium cząstek: klasyczny gaz i rozpady z tablic PDG.
//!
//! # Czym to jest
//!
//! Klasyczna symulacja gazu. Każda cząstka ma gatunek z [`particles`] — masę,
//! ładunek, spin, kolor, liczby kwantowe i czas życia z tablic PDG. Na cząstki
//! działają siły ([`forces`]), całkowane relatywistycznie na pędzie, a nad tym
//! siedzi warstwa stochastyczna ([`decays`]), która rozpada cząstki i anihiluje
//! pary.
//!
//! # Czym to NIE jest
//!
//! **To nie jest kwantowa teoria pola.** Model Standardowy jest teorią pola; ta
//! symulacja liczy klasyczne trajektorie punktów. Różnica nie jest kwestią
//! dokładności, tylko rodzaju opisu, i żadne zagęszczenie kroku jej nie zasypie.
//! Konkretnie brakuje:
//!
//! - **stanów związanych**. Klasyczny elektron w polu protonu spada na jądro —
//!   atom wodoru nie istnieje bez mechaniki kwantowej. Zmiękczenie potencjału
//!   ([`ForceConfig::softening`]) powstrzymuje ten upadek i jest **protezą za
//!   kwantowanie**, a nie efektem fizycznym. Diagnostyka pokazuje, ile energii
//!   z niego wynika;
//! - **amplitud i interferencji**. Rozpad jest losowaniem ze współczynników
//!   rozgałęzienia, a rozkłady kątowe są izotropowe w układzie spoczynkowym. Widmo
//!   Michela w rozpadzie mionu, korelacje spinowe i naruszenie parzystości są poza
//!   tym modelem;
//! - **hadronizacji**. Uwolniony kwark w naturze wytwarza nowe pary i ubiera się
//!   w hadrony. Tutaj czuje tylko liniowy człon potencjału Cornella, czyli
//!   uwięzienie bez zerwania struny;
//! - **rozpraszania z przekazem pędu innego niż z sił**. Nie ma zderzeń twardych,
//!   promieniowania hamowania ani produkcji par z pola.
//!
//! Poza tą listą model liczy to, co obiecuje, i **mierzy** własne ograniczenia:
//! ładunek, liczba barionowa i liczby leptonowe są zachowywane **dokładnie**
//! (są liczbami całkowitymi), a bilans energii i pędu przy rozpadzie jest
//! sprawdzany, a nie zakładany.
//!
//! # Dwie skale czasu, które się nie spotykają
//!
//! Siły działają na skali femtometra, czyli `~1 fm/c`. Mion żyje `6,6·10¹⁷ fm/c`.
//! Te dwie liczby dzieli siedemnaście rzędów wielkości i **żaden krok całkowania
//! nie obsłuży obu naraz**. Nie jest to usterka do naprawienia, tylko własność
//! przyrody: na czas, w którym dzieje się oddziaływanie, rozpad jest niemożliwy,
//! a na czas, w którym dochodzi do rozpadu, cząstka jest dawno sama.
//!
//! Silnik nie udaje, że tego problemu nie ma. Każdy zestaw nastaw wybiera skalę
//! kroku właściwą dla zjawiska, które bada, a [`diagnostics`] wypisuje obie skale
//! i ostrzega, gdy krok jest tak dobrany, że drugi rodzaj zjawiska jest niewidoczny.

pub mod config;
pub mod decays;
pub mod diagnostics;
pub mod engine;
pub mod forces;
pub mod kinematics;
pub mod particles;
pub mod presets;
pub mod spawn;
pub mod state;
pub mod units;

pub use config::{Config, ForceConfig};
pub use diagnostics::Snapshot;
pub use engine::Engine;
pub use particles::{Family, Particle, Species};
pub use state::State;
