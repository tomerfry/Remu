//! Interrupt vectors and signal modelling.
//!
//! Interrupts are CPU *input signals*, not memory operations, so they live here
//! rather than on the [`Bus`](crate::bus::Bus) trait. The CPU polls them at
//! instruction boundaries (see [`Cpu::step`](crate::cpu::Cpu::step)).
//!
//! - **NMI** is edge-triggered: latched on a high→low transition, cleared when
//!   serviced. Modelled by [`Cpu::set_nmi`](crate::cpu::Cpu::set_nmi) (tracks the
//!   previous level) or [`Cpu::trigger_nmi`](crate::cpu::Cpu::trigger_nmi).
//! - **IRQ** is level-triggered and maskable: serviced between instructions only
//!   while the line is asserted and the `I` flag is clear.
//! - **RESET** loads `PC` from the reset vector (see
//!   [`Cpu::reset`](crate::cpu::Cpu::reset)).

/// Vector address for non-maskable interrupts (`$FFFA`/`$FFFB`).
pub const NMI_VECTOR: u16 = 0xFFFA;

/// Vector address for reset (`$FFFC`/`$FFFD`).
pub const RESET_VECTOR: u16 = 0xFFFC;

/// Vector address for maskable interrupts and `BRK` (`$FFFE`/`$FFFF`).
pub const IRQ_VECTOR: u16 = 0xFFFE;

/// Number of cycles consumed when servicing an interrupt (NMI/IRQ) or reset.
pub const INTERRUPT_CYCLES: u8 = 7;
