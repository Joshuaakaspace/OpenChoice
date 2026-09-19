//! Run the device console on a host, against a real catalog.
//!
//! The firmware is ~80 lines of serial plumbing around this same `Console`,
//! so this is the fast way to develop and demo the on-device experience
//! without a board in hand — and the honest way to show its output.
//!
//! ```sh
//! cargo run -p openchoice-embedded --example repl -- catalog/openchoice-tiny.ocb
//! echo "gpu rtx 4090
//! vram 24576
//! go" | cargo run -p openchoice-embedded --example repl -- catalog/openchoice-tiny.ocb
//! ```

use std::io::{BufRead, Write};

use openchoice_core::Catalog;
use openchoice_embedded::{Console, Outcome, ReportBuf};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "catalog/openchoice.ocb".into());
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(1);
        }
    };
    let catalog = match Catalog::parse(&bytes) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{path} is not a valid catalog: {e:?}");
            std::process::exit(1);
        }
    };

    // The same buffer size the firmware uses, so output truncates here exactly
    // where it would truncate on the device.
    let mut report: ReportBuf<2048> = ReportBuf::new();
    let mut console = Console::new();

    println!(
        "OpenChoice — {} models in {} KB of flash",
        catalog.len(),
        bytes.len() / 1024
    );
    println!("type `help`, or `go` to rank for the default machine\n");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    loop {
        print!("openchoice> ");
        let _ = stdout.flush();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }

        if console.handle(line.trim_end(), &catalog, &mut report) != Outcome::Empty {
            println!("{}", report.as_str());
        }
    }
}
