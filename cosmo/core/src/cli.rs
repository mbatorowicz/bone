//! Bieg bez okna: ten sam silnik, sterowany argumentami wiersza poleceń.
//!
//! Istnieje po to, żeby długi bieg dał się puścić na noc i zostawił po sobie pliki,
//! a nie okno, którego nie wolno zamknąć. Dlatego jedyne, co CLI robi ponad panel,
//! to zapis: checkpoint do wznowienia i trajektoria do późniejszego oglądania.
//!
//! Rozbiór argumentów jest rozdzielony od wykonania ([`parse`] kontra [`execute`]).
//! Bez tego rozdziału sprawdzenie, że `--steps abc` daje czytelny błąd, wymagałoby
//! policzenia symulacji.

use std::path::PathBuf;

use crate::lcdm;
use crate::session::Session;
use crate::sr;

const HELP: &str = "\
bone — grawitacja N ciał

    bone                          okno z panelem (domyślnie)
    bone sr      [opcje]          odosobniona chmura, kinematyka SR
    bone lcdm    [opcje]          próbka wszechświata ΛCDM (Planck 2018)
    bone presety                  wypisz nazwy zestawów nastaw SR
    bone --pomoc                  ten opis

Opcje wspólne:
    --kroki N                     ile kroków policzyć (0 = bez limitu dla SR)
    --do KATALOG                  gdzie zapisać wynik
    --cicho                       nie wypisuj pomiarów w trakcie

Opcje trybu sr:
    --zestaw NAZWA                zestaw nastaw (patrz `bone presety`)
    --config PLIK                 konfiguracja z pliku JSON
    --czastek N                   nadpisz liczbę cząstek
    --wznow                       wznów z checkpointu w katalogu wyjściowym

Opcje trybu lcdm:
    --zestaw NAZWA                liniowy | struktury | struktury64
    --siatka N                    bok siatki warunków początkowych
    --probka L                    bok próbki w Mpc/h
    --z-koniec Z                  przesunięcie ku czerwieni, na którym skończyć
    --wznow                       wznów z checkpointu w katalogu wyjściowym
";

#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    Gui,
    Help,
    ListPresets,
    Relativistic(RelativisticJob),
    Cosmological(CosmologicalJob),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shared {
    pub steps: u64,
    pub out_dir: PathBuf,
    pub quiet: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RelativisticJob {
    pub shared: Shared,
    pub preset: Option<String>,
    pub config_file: Option<PathBuf>,
    pub particles: Option<usize>,
    pub resume: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CosmologicalJob {
    pub shared: Shared,
    pub preset: Option<String>,
    pub n_grid: Option<usize>,
    pub box_size: Option<f64>,
    pub z_end: Option<f64>,
    pub resume: bool,
}

/// Rozbierz argumenty (bez nazwy programu).
///
/// # Errors
/// Gdy podana jest nieznana opcja albo wartość, której nie da się odczytać. Cichej
/// tolerancji nie ma celowo: literówka w nazwie opcji znaczyłaby bieg z innymi
/// parametrami niż zamierzone, co przy biegu na noc jest kosztowną pomyłką.
pub fn parse(args: &[String]) -> Result<Command, String> {
    let Some(first) = args.first() else {
        return Ok(Command::Gui);
    };
    match first.as_str() {
        "--pomoc" | "--help" | "-h" => return Ok(Command::Help),
        "presety" | "presets" => return Ok(Command::ListPresets),
        _ => {}
    }

    let mut shared = Shared {
        steps: 0,
        out_dir: PathBuf::from("runs/latest"),
        quiet: false,
    };
    let mut preset = None;
    let mut config_file = None;
    let mut particles = None;
    let mut n_grid = None;
    let mut box_size = None;
    let mut z_end = None;
    let mut resume = false;

    let mut rest = args[1..].iter();
    while let Some(flag) = rest.next() {
        let mut value = || -> Result<String, String> {
            rest.next()
                .cloned()
                .ok_or_else(|| format!("opcja {flag} wymaga wartości"))
        };
        match flag.as_str() {
            "--kroki" => shared.steps = number(&value()?, flag)?,
            "--do" => shared.out_dir = PathBuf::from(value()?),
            "--cicho" => shared.quiet = true,
            "--zestaw" => preset = Some(value()?),
            "--config" => config_file = Some(PathBuf::from(value()?)),
            "--czastek" => particles = Some(number(&value()?, flag)?),
            "--wznow" => resume = true,
            "--siatka" => n_grid = Some(number(&value()?, flag)?),
            "--probka" => box_size = Some(number(&value()?, flag)?),
            "--z-koniec" => z_end = Some(number(&value()?, flag)?),
            other => return Err(format!("nieznana opcja: {other}")),
        }
    }

    match first.as_str() {
        "sr" => {
            reject_foreign(&[
                ("--siatka", n_grid.is_some()),
                ("--probka", box_size.is_some()),
                ("--z-koniec", z_end.is_some()),
            ])?;
            Ok(Command::Relativistic(RelativisticJob {
                shared,
                preset,
                config_file,
                particles,
                resume,
            }))
        }
        "lcdm" => {
            reject_foreign(&[
                ("--config", config_file.is_some()),
                ("--czastek", particles.is_some()),
            ])?;
            Ok(Command::Cosmological(CosmologicalJob {
                shared,
                preset,
                n_grid,
                box_size,
                z_end,
                resume,
            }))
        }
        other => Err(format!("nieznane polecenie: {other}")),
    }
}

/// Opcja z innego trybu jest błędem, nie życzeniem do zignorowania.
///
/// `bone lcdm --czastek 8000` wygląda, jakby ustawiał liczbę cząstek, a ustawia ją
/// `--siatka`. Milczące przyjęcie takiego argumentu dałoby bieg o innej wielkości niż
/// zamówiona i nic by o tym nie powiedziało.
fn reject_foreign(flags: &[(&str, bool)]) -> Result<(), String> {
    match flags.iter().find(|(_, present)| *present) {
        Some((flag, _)) => Err(format!("opcja {flag} nie należy do tego trybu")),
        None => Ok(()),
    }
}

fn number<T: std::str::FromStr>(text: &str, flag: &str) -> Result<T, String> {
    text.parse()
        .map_err(|_| format!("{flag}: nie umiem odczytać liczby z „{text}”"))
}

/// # Errors
/// Gdy nie da się wczytać konfiguracji, zapisać wyniku albo bieg się rozbiegnie.
pub fn execute(command: Command) -> Result<(), String> {
    match command {
        Command::Gui => Err("okno jest poza bone-core — uruchom binarkę bez argumentów".into()),
        Command::Help => {
            print!("{HELP}");
            Ok(())
        }
        Command::ListPresets => {
            for id in sr::presets::ids() {
                println!("{id}");
            }
            Ok(())
        }
        Command::Relativistic(job) => run_relativistic(job),
        Command::Cosmological(job) => run_cosmological(job),
    }
}

fn run_relativistic(job: RelativisticJob) -> Result<(), String> {
    let out = job.shared.out_dir.clone();
    let mut session = if job.resume {
        Session::resume(out, true)?
    } else {
        Session::start_sr(build_sr_config(&job)?, out, true)?
    };

    // Zero znaczy „licz, dopóki nie przerwę" — sensowne tylko interaktywnie, więc
    // w trybie wsadowym dostaje skończony limit zamiast pętli bez końca.
    let limit = if job.shared.steps > 0 {
        job.shared.steps
    } else if job.resume {
        2_000
    } else {
        build_sr_config(&job)?.run.steps.max(2_000)
    };

    for _ in 1..=limit {
        let report = session.advance(1)?;
        if let Some(hint) = session.accuracy_hint() {
            eprintln!("uwaga: {hint}");
        }
        if let Some(snapshot) = session.sr_snapshot() {
            let every = session.diagnostics_every();
            if snapshot.step.is_multiple_of(every) && !job.shared.quiet {
                let error = match snapshot.force_error {
                    Some(e) => format!("  błąd siły={:>5.2}%", 100.0 * e.rms),
                    None => String::new(),
                };
                println!(
                    "krok {:>7}  t={:>10.4}  E={:>11.4e}  dryf={:>+9.2e}  2K/|U|={:>6.3}  β_max={:.3}{error}",
                    snapshot.step,
                    snapshot.time,
                    snapshot.total_energy,
                    snapshot.energy_drift,
                    snapshot.virial,
                    snapshot.beta_max
                );
            }
        }
        for warning in report.warnings {
            eprintln!("ostrzeżenie: {warning}");
        }
    }

    finish_session(&mut session, job.shared.quiet)
}

fn build_sr_config(job: &RelativisticJob) -> Result<sr::Config, String> {
    let mut cfg = match (&job.config_file, &job.preset) {
        (Some(path), _) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("nie mogę wczytać {}: {e}", path.display()))?;
            sr::Config::from_json(&text).map_err(|e| format!("{}: {e}", path.display()))?
        }
        (None, Some(name)) => sr::presets::preset(name).ok_or_else(|| {
            format!(
                "nie znam zestawu „{name}”; dostępne: {}",
                sr::presets::ids().join(", ")
            )
        })?,
        (None, None) => sr::presets::galaxy(),
    };
    if let Some(n) = job.particles {
        cfg.spawn.n_particles = n.max(2);
    }
    Ok(cfg)
}

fn run_cosmological(job: CosmologicalJob) -> Result<(), String> {
    let out = job.shared.out_dir.clone();
    let mut session = if job.resume {
        Session::resume(out, true)?
    } else {
        Session::start_lcdm(build_lcdm_config(&job)?, out, true)?
    };

    let limit = if job.shared.steps > 0 {
        job.shared.steps
    } else {
        u64::MAX
    };
    for _ in 0..limit {
        if session.finished() {
            break;
        }
        session.advance(1)?;
        if let Some((step, z, age, t, w, li)) = session.lcdm_log_values() {
            if step.is_multiple_of(20) && !job.shared.quiet {
                println!(
                    "krok {:>7}  z={:>8.3}  wiek={:>6.2} Gyr  T={:>11.4e}  W={:>11.4e}  LI={:>+9.2e}",
                    step, z, age, t, w, li
                );
            }
        }
    }

    finish_session(&mut session, job.shared.quiet)
}

fn build_lcdm_config(job: &CosmologicalJob) -> Result<lcdm::RunConfig, String> {
    let mut cfg = match job.preset.as_deref() {
        None => lcdm::Preset::Structure.config(),
        Some(name) => lcdm::presets::by_id(name).ok_or_else(|| {
            format!(
                "nie znam zestawu „{name}”; dostępne: {}",
                lcdm::presets::ids().join(", ")
            )
        })?,
    };
    if let Some(n) = job.n_grid {
        cfg.n_grid = n.max(8);
        cfg.pm_grid = cfg.n_grid.min(64);
    }
    if let Some(l) = job.box_size {
        cfg.box_size = l;
    }
    if let Some(z) = job.z_end {
        cfg.z_end = z;
    }
    Ok(cfg)
}

fn finish_session(session: &mut Session, quiet: bool) -> Result<(), String> {
    let frames = session.flush_recording()?;
    let path = session
        .save_checkpoint()
        .map_err(|e| format!("zapis checkpointu: {e}"))?;
    if !quiet {
        println!("checkpoint: {}", path.display());
        println!("klatek: {frames}");
        if let Some((_, z, age, _, _, _)) = session.lcdm_log_values() {
            println!("koniec: z={z:.3}  wiek={age:.2} Gyr");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(text: &str) -> Vec<String> {
        text.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn no_arguments_opens_the_window() {
        assert_eq!(parse(&[]).unwrap(), Command::Gui);
    }

    #[test]
    fn help_and_presets_are_recognized() {
        for flag in ["--pomoc", "--help", "-h"] {
            assert_eq!(parse(&args(flag)).unwrap(), Command::Help);
        }
        assert_eq!(parse(&args("presety")).unwrap(), Command::ListPresets);
    }

    #[test]
    fn relativistic_job_collects_its_options() {
        let cmd = parse(&args(
            "sr --zestaw collapse --kroki 500 --do wyniki/a --czastek 8000 --cicho",
        ))
        .unwrap();
        let Command::Relativistic(job) = cmd else {
            panic!("zły tryb: {cmd:?}");
        };
        assert_eq!(job.preset.as_deref(), Some("collapse"));
        assert_eq!(job.shared.steps, 500);
        assert_eq!(job.shared.out_dir, PathBuf::from("wyniki/a"));
        assert_eq!(job.particles, Some(8_000));
        assert!(job.shared.quiet);
        assert!(!job.resume);
    }

    #[test]
    fn cosmological_job_collects_its_options() {
        let cmd = parse(&args(
            "lcdm --zestaw liniowy --siatka 32 --probka 150.5 --z-koniec 0.5",
        ))
        .unwrap();
        let Command::Cosmological(job) = cmd else {
            panic!("zły tryb: {cmd:?}");
        };
        assert_eq!(job.preset.as_deref(), Some("liniowy"));
        assert_eq!(job.n_grid, Some(32));
        assert_eq!(job.box_size, Some(150.5));
        assert_eq!(job.z_end, Some(0.5));
        assert!(!job.resume);
    }

    #[test]
    fn defaults_are_filled_in() {
        let Command::Relativistic(job) = parse(&args("sr")).unwrap() else {
            panic!();
        };
        assert_eq!(job.shared.steps, 0);
        assert_eq!(job.shared.out_dir, PathBuf::from("runs/latest"));
        assert!(!job.shared.quiet);
        assert!(job.preset.is_none());
    }

    /// Literówka w nazwie opcji MUSI być błędem. Przemilczenie jej znaczyłoby bieg
    /// z domyślnymi parametrami zamiast zamówionych — i nikt by tego nie zauważył
    /// przed obejrzeniem wyniku.
    #[test]
    fn unknown_flags_and_commands_are_errors() {
        assert!(parse(&args("sr --zestaww galaxy")).is_err());
        assert!(parse(&args("polataj")).is_err());
    }

    #[test]
    fn missing_and_malformed_values_are_errors() {
        let err = parse(&args("sr --kroki")).unwrap_err();
        assert!(err.contains("wymaga wartości"), "{err}");
        let err = parse(&args("sr --kroki dużo")).unwrap_err();
        assert!(err.contains("nie umiem odczytać"), "{err}");
        assert!(parse(&args("lcdm --probka nie")).is_err());
    }

    #[test]
    fn options_from_the_other_mode_are_rejected() {
        let err = parse(&args("lcdm --czastek 8000")).unwrap_err();
        assert!(err.contains("--czastek"), "{err}");
        let err = parse(&args("sr --siatka 32")).unwrap_err();
        assert!(err.contains("--siatka"), "{err}");
        // Opcje wspólne muszą przechodzić w obu trybach.
        assert!(parse(&args("sr --kroki 5 --do x --cicho")).is_ok());
        assert!(parse(&args("lcdm --kroki 5 --do x --cicho")).is_ok());
    }

    #[test]
    fn resume_flag_takes_no_value() {
        let Command::Relativistic(job) = parse(&args("sr --wznow --kroki 10")).unwrap() else {
            panic!();
        };
        assert!(job.resume);
        assert_eq!(job.shared.steps, 10);
        let Command::Cosmological(job) = parse(&args("lcdm --wznow")).unwrap() else {
            panic!();
        };
        assert!(job.resume);
    }

    #[test]
    fn unknown_preset_names_are_reported_with_the_list() {
        let job = RelativisticJob {
            shared: Shared {
                steps: 1,
                out_dir: PathBuf::from("x"),
                quiet: true,
            },
            preset: Some("nie-ma-takiego".to_string()),
            config_file: None,
            particles: None,
            resume: false,
        };
        let err = match build_sr_config(&job) {
            Err(message) => message,
            Ok(_) => panic!("nieznany zestaw został przyjęty"),
        };
        assert!(err.contains("galaxy"), "brak listy w komunikacie: {err}");
    }

    #[test]
    fn help_text_mentions_every_command() {
        for keyword in ["sr", "lcdm", "presety", "--kroki", "--wznow"] {
            assert!(HELP.contains(keyword), "brak {keyword} w pomocy");
        }
    }
}
