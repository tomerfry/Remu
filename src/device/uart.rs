//! A minimal 16550-compatible UART for polled serial I/O.
//!
//! Just enough of the register file to be a console: `THR` writes go to an
//! output sink (stdout by default), `RBR` reads drain a host-fed input queue,
//! and `LSR` reports transmit-empty and receive-data-ready. Transmission is
//! instantaneous — there is no FIFO, baud rate, or modem emulation — so `LSR`
//! always reports the transmitter empty.
//!
//! The remaining registers are inert storage so guest init code runs
//! unsurprised: `IER`, `LCR`, `MCR` and scratch read back what was written,
//! and the divisor latch (offsets 0/1 while the `LCR` DLAB bit is set) is
//! stored without affecting timing or leaking into the output.
//!
//! Interrupts are not wired up yet: `IER` is stored but never acted on, and
//! `IIR` always reads "no interrupt pending". When the [`Device`] trait grows
//! an IRQ line (see its docs), this UART asserts it from `IER` and the same
//! rx/tx state used by `LSR` today — no register model changes needed.

use std::collections::VecDeque;
use std::io;

use super::Device;

/// Receiver Buffer Register (read) / Transmitter Holding Register (write).
/// With the `LCR` DLAB bit set, this is the divisor latch low byte instead.
pub const RBR_THR: u16 = 0;
/// Interrupt Enable Register. With DLAB set, the divisor latch high byte.
pub const IER: u16 = 1;
/// Interrupt Identification Register (read) / FIFO Control Register (write).
pub const IIR_FCR: u16 = 2;
/// Line Control Register. Bit 7 is DLAB, the divisor latch access bit.
pub const LCR: u16 = 3;
/// Modem Control Register.
pub const MCR: u16 = 4;
/// Line Status Register (read-only).
pub const LSR: u16 = 5;
/// Modem Status Register (read-only).
pub const MSR: u16 = 6;
/// Scratch Register.
pub const SCR: u16 = 7;

/// `LSR` bit 0: data ready — a received byte is waiting in `RBR`.
pub const LSR_DATA_READY: u8 = 0x01;
/// `LSR` bit 5: transmitter holding register empty (`THR` can be written).
pub const LSR_THR_EMPTY: u8 = 0x20;
/// `LSR` bit 6: transmitter idle (holding and shift registers both empty).
pub const LSR_TX_IDLE: u8 = 0x40;

/// `LCR` bit 7: while set, offsets 0/1 address the divisor latch.
const LCR_DLAB: u8 = 0x80;

/// A minimal 16550 UART. See the [module docs](self) for what is and isn't
/// emulated.
pub struct Uart16550 {
    /// Where transmitted bytes go.
    tx: Box<dyn io::Write>,
    /// Host-fed bytes waiting to be read through `RBR`.
    rx: VecDeque<u8>,
    /// Inert storage for the stub registers (see module docs).
    ier: u8,
    lcr: u8,
    mcr: u8,
    scratch: u8,
    /// Divisor latch, `[DLL, DLM]` — stored, never used for timing.
    divisor: [u8; 2],
}

impl Uart16550 {
    /// Size of the register window in bytes (offsets 0–7).
    pub const WINDOW: u16 = 8;

    /// Create a UART transmitting to stdout.
    pub fn new() -> Self {
        Uart16550::with_output(Box::new(io::stdout()))
    }

    /// Create a UART transmitting to `tx` (lets tests capture the output).
    pub fn with_output(tx: Box<dyn io::Write>) -> Self {
        Uart16550 {
            tx,
            rx: VecDeque::new(),
            ier: 0,
            lcr: 0,
            mcr: 0,
            scratch: 0,
            divisor: [0, 0],
        }
    }

    /// Queue `bytes` for the guest to read through `RBR`.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.rx.extend(bytes);
    }

    fn dlab(&self) -> bool {
        self.lcr & LCR_DLAB != 0
    }
}

impl Default for Uart16550 {
    fn default() -> Self {
        Uart16550::new()
    }
}

impl Device for Uart16550 {
    fn read(&mut self, offset: u16) -> u8 {
        match offset {
            RBR_THR if self.dlab() => self.divisor[0],
            // Reading with nothing buffered yields 0 (real hardware would
            // return the stale previous byte).
            RBR_THR => self.rx.pop_front().unwrap_or(0),
            IER if self.dlab() => self.divisor[1],
            IER => self.ier,
            // No interrupt sources yet, so IIR always reads "none pending".
            IIR_FCR => 0x01,
            LCR => self.lcr,
            MCR => self.mcr,
            LSR => {
                // Transmission is instantaneous, so TX is always empty.
                let rx_ready = if self.rx.is_empty() {
                    0
                } else {
                    LSR_DATA_READY
                };
                LSR_THR_EMPTY | LSR_TX_IDLE | rx_ready
            }
            MSR => 0,
            SCR => self.scratch,
            // Unreachable through an 8-byte window; open bus otherwise.
            _ => 0,
        }
    }

    fn write(&mut self, offset: u16, value: u8) {
        match offset {
            RBR_THR if self.dlab() => self.divisor[0] = value,
            RBR_THR => {
                // Bus writes are infallible, so a broken sink drops bytes
                // rather than failing. Flush per byte to keep an interactive
                // console responsive.
                let _ = self.tx.write_all(&[value]);
                let _ = self.tx.flush();
            }
            IER if self.dlab() => self.divisor[1] = value,
            IER => self.ier = value,
            // FCR: no FIFOs to configure.
            IIR_FCR => {}
            LCR => self.lcr = value,
            MCR => self.mcr = value,
            // Read-only status registers.
            LSR | MSR => {}
            SCR => self.scratch = value,
            _ => {}
        }
    }
}
