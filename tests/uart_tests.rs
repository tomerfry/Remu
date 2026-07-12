//! Tests for the 16550 UART and the [`SystemBus`] address decoder, ending with
//! a guest 6502 program that prints through the UART by polling `LSR`.

use std::cell::RefCell;
use std::io;
use std::rc::Rc;

use remu::Cpu;
use remu::bus::Bus;
use remu::device::uart::{self, LSR_DATA_READY, LSR_THR_EMPTY, LSR_TX_IDLE, Uart16550};
use remu::device::{Device, SystemBus};

/// Where the UART's 8-byte register window lives in the guest address space.
const UART_BASE: u16 = 0xF000;

/// An `io::Write` sink the test can still read after boxing it into the UART.
#[derive(Clone, Default)]
struct SharedBuf(Rc<RefCell<Vec<u8>>>);

impl SharedBuf {
    fn contents(&self) -> Vec<u8> {
        self.0.borrow().clone()
    }
}

impl io::Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A UART capturing its transmit output, plus a handle to read it back.
fn uart_with_captured_output() -> (Uart16550, SharedBuf) {
    let out = SharedBuf::default();
    (Uart16550::with_output(Box::new(out.clone())), out)
}

#[test]
fn thr_transmits_string_byte_by_byte() {
    let (mut uart, out) = uart_with_captured_output();
    for &byte in b"Hello, UART!" {
        uart.write(uart::RBR_THR, byte);
    }
    assert_eq!(out.contents(), b"Hello, UART!");
}

#[test]
fn lsr_reports_tx_empty_and_rx_data_ready() {
    let (mut uart, _out) = uart_with_captured_output();

    // Nothing buffered: transmitter empty, no data ready.
    assert_eq!(uart.read(uart::LSR), LSR_THR_EMPTY | LSR_TX_IDLE);

    uart.feed(b"hi");
    assert_eq!(
        uart.read(uart::LSR),
        LSR_THR_EMPTY | LSR_TX_IDLE | LSR_DATA_READY
    );

    // RBR drains the queue in order; data-ready drops when it empties.
    assert_eq!(uart.read(uart::RBR_THR), b'h');
    assert_eq!(uart.read(uart::RBR_THR), b'i');
    assert_eq!(uart.read(uart::LSR), LSR_THR_EMPTY | LSR_TX_IDLE);
    assert_eq!(uart.read(uart::RBR_THR), 0); // empty queue reads as 0
}

#[test]
fn dlab_redirects_divisor_latch_without_transmitting() {
    let (mut uart, out) = uart_with_captured_output();

    // With DLAB set, offsets 0/1 are the divisor latch, not THR/IER.
    uart.write(uart::LCR, 0x80);
    uart.write(uart::RBR_THR, 0x0C);
    uart.write(uart::IER, 0x03);
    assert_eq!(uart.read(uart::RBR_THR), 0x0C);
    assert_eq!(uart.read(uart::IER), 0x03);
    assert_eq!(
        out.contents(),
        b"",
        "divisor writes must not reach the output"
    );

    // DLAB cleared: offset 0 transmits again and IER is back.
    uart.write(uart::LCR, 0x00);
    assert_eq!(uart.read(uart::IER), 0x00);
    uart.write(uart::RBR_THR, b'x');
    assert_eq!(out.contents(), b"x");
}

#[test]
fn stub_registers_read_and_write_without_surprises() {
    let (mut uart, _out) = uart_with_captured_output();

    // Read/write storage round-trips.
    uart.write(uart::IER, 0x0F);
    uart.write(uart::LCR, 0x03);
    uart.write(uart::MCR, 0x0B);
    uart.write(uart::SCR, 0x5A);
    assert_eq!(uart.read(uart::IER), 0x0F);
    assert_eq!(uart.read(uart::LCR), 0x03);
    assert_eq!(uart.read(uart::MCR), 0x0B);
    assert_eq!(uart.read(uart::SCR), 0x5A);

    // IIR reads "no interrupt pending"; FCR writes are accepted and ignored.
    uart.write(uart::IIR_FCR, 0xC7);
    assert_eq!(uart.read(uart::IIR_FCR), 0x01);

    // Status registers are read-only.
    uart.write(uart::LSR, 0x00);
    uart.write(uart::MSR, 0xFF);
    assert_eq!(uart.read(uart::LSR), LSR_THR_EMPTY | LSR_TX_IDLE);
    assert_eq!(uart.read(uart::MSR), 0x00);
}

#[test]
fn system_bus_routes_window_and_falls_back_to_ram() {
    let (mut uart, out) = uart_with_captured_output();
    uart.feed(b"A");

    let mut bus = SystemBus::new();
    bus.map(UART_BASE, Uart16550::WINDOW, Box::new(uart));

    // Inside the window: device registers, not RAM.
    assert_eq!(
        bus.read(UART_BASE + uart::LSR),
        LSR_THR_EMPTY | LSR_TX_IDLE | LSR_DATA_READY
    );
    assert_eq!(bus.read(UART_BASE + uart::RBR_THR), b'A');
    bus.write(UART_BASE + uart::RBR_THR, b'B');
    assert_eq!(out.contents(), b"B");

    // Outside the window (including both neighbors): plain RAM.
    for addr in [0x0000, 0x1234, UART_BASE - 1, UART_BASE + Uart16550::WINDOW] {
        bus.write(addr, 0x42);
        assert_eq!(bus.read(addr), 0x42);
        assert_eq!(bus.ram.ram[addr as usize], 0x42);
    }
    assert_eq!(out.contents(), b"B", "RAM writes must not reach the UART");
}

/// Guest program: print a zero-terminated message through the UART by polling
/// `LSR` for transmit-empty before each `THR` write, then jam the CPU (KIL) so
/// the host loop knows it finished.
///
/// ```text
/// 0600: A2 00      LDX #$00
/// loop:
/// 0602: AD 05 F0   LDA $F005     ; LSR
/// 0605: 29 20      AND #$20      ; transmit holding register empty?
/// 0607: F0 F9      BEQ loop      ; no -> keep polling
/// 0609: BD 15 06   LDA $0615,X   ; msg[X]
/// 060C: F0 06      BEQ done      ; zero terminator -> done
/// 060E: 8D 00 F0   STA $F000     ; THR
/// 0611: E8         INX
/// 0612: D0 EE      BNE loop
/// done:
/// 0614: 02         KIL
/// msg:
/// 0615: "Hello from guest", 00
/// ```
#[test]
fn guest_program_prints_hello_from_guest() {
    #[rustfmt::skip]
    let mut program = vec![
        0xA2, 0x00,             // LDX #$00
        0xAD, 0x05, 0xF0,       // LDA $F005
        0x29, 0x20,             // AND #$20
        0xF0, 0xF9,             // BEQ $0602
        0xBD, 0x15, 0x06,       // LDA $0615,X
        0xF0, 0x06,             // BEQ $0614
        0x8D, 0x00, 0xF0,       // STA $F000
        0xE8,                   // INX
        0xD0, 0xEE,             // BNE $0602
        0x02,                   // KIL
    ];
    program.extend_from_slice(b"Hello from guest\0");

    let (uart, out) = uart_with_captured_output();
    let mut bus = SystemBus::new();
    bus.map(UART_BASE, Uart16550::WINDOW, Box::new(uart));
    bus.ram.load(0x0600, &program);
    bus.ram.set_reset_vector(0x0600);

    let mut cpu = Cpu::new();
    cpu.reset(&mut bus);
    for _ in 0..100_000 {
        if cpu.halted {
            break;
        }
        cpu.step(&mut bus);
    }

    assert!(cpu.halted, "guest program did not reach its final KIL");
    assert_eq!(out.contents(), b"Hello from guest");
}
