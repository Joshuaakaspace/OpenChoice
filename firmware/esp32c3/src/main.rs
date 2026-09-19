//! OpenChoice on an ESP32-C3: an offline LLM fit oracle you can hold.
//!
//! The device holds the whole model catalog in flash and answers "will this
//! run on that machine?" with no network, no host, and no allocator. Plug it
//! into USB, open a serial monitor, and type.
//!
//! ```text
//! openchoice> gpu rtx 4090
//! RTX 4090: 1008 GB/s, 165.4 TFLOPS fp16
//! openchoice> vram 24576
//! openchoice> ram 65536
//! openchoice> go
//! ```
//!
//! Everything interesting lives in `openchoice-embedded`, which is plain
//! `no_std` Rust with no hardware dependency at all — this file is only the
//! serial plumbing, which is why the logic can be unit-tested on a laptop.

#![no_std]
#![no_main]

use core::fmt::Write as _;

use esp_backtrace as _;
use esp_hal::main;
use esp_hal::usb::usb_serial_jtag::UsbSerialJtag;
use openchoice_core::Catalog;
use openchoice_embedded::{Console, Outcome, ReportBuf};

/// The packed catalog, linked straight into flash.
///
/// Read in place: nothing is ever copied into the 400 KB of SRAM this chip
/// has. Build a smaller one with `--top N` if your flash partition is tight.
static CATALOG: &[u8] = include_bytes!("../../../catalog/openchoice-tiny.ocb");

/// One line of typed input. Longer lines are truncated rather than wrapped,
/// which is the right trade for a hand-typed console.
const LINE_MAX: usize = 96;
/// Report buffer. Sized to hold a full ranking table; the largest single
/// allocation in the firmware, and it is a static-lifetime stack array.
const REPORT_MAX: usize = 2048;

#[main]
fn main() -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default());
    let mut serial = UsbSerialJtag::new(peripherals.USB_DEVICE);

    let catalog = match Catalog::parse(CATALOG) {
        Ok(c) => c,
        Err(_) => {
            // A bad catalog is not recoverable, but it must be legible: a
            // silent hang here looks identical to dead hardware.
            loop {
                let _ = writeln!(serial, "fatal: embedded catalog failed to parse");
                delay_ms(2000);
            }
        }
    };

    let mut console = Console::new();
    let mut report: ReportBuf<REPORT_MAX> = ReportBuf::new();

    let _ = writeln!(
        serial,
        "\r\nOpenChoice — {} models in {} KB of flash\r\ntype `help`, or `go` to rank for the default machine\r\n",
        catalog.len(),
        CATALOG.len() / 1024
    );

    let mut line = [0u8; LINE_MAX];
    let mut len = 0usize;
    let _ = write!(serial, "openchoice> ");

    loop {
        let byte = match serial.read_byte() {
            Ok(b) => b,
            Err(_) => continue,
        };

        match byte {
            b'\r' | b'\n' => {
                let _ = write!(serial, "\r\n");
                let text = core::str::from_utf8(&line[..len]).unwrap_or("");
                match console.handle(text, &catalog, &mut report) {
                    Outcome::Empty => {}
                    _ => {
                        // The report holds \n line breaks; a serial terminal
                        // wants \r\n, so translate on the way out rather than
                        // teaching the renderer about terminals.
                        for chunk in report.as_str().split('\n') {
                            let _ = write!(serial, "{chunk}\r\n");
                        }
                    }
                }
                len = 0;
                let _ = write!(serial, "openchoice> ");
            }
            // Backspace and delete.
            0x08 | 0x7F => {
                if len > 0 {
                    len -= 1;
                    let _ = write!(serial, "\x08 \x08");
                }
            }
            b if b.is_ascii_graphic() || b == b' ' => {
                if len < LINE_MAX {
                    line[len] = b;
                    len += 1;
                    let _ = serial.write(&[b]);
                }
            }
            _ => {}
        }
    }
}

/// Crude busy-wait. Only used on the fatal path, where precision is
/// irrelevant and pulling in a timer peripheral is not worth it.
fn delay_ms(ms: u32) {
    // The C3 core runs at 160 MHz; this is deliberately approximate.
    let iterations = ms as u64 * 16_000;
    for _ in 0..iterations {
        core::hint::spin_loop();
    }
}
