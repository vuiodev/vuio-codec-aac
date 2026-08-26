//! Bitstream I/O and Cyclic Redundancy Check (CRC) Subsystems

pub mod crc;
pub mod reader;
pub mod writer;

pub use crc::{crc16_adts, crc16_sbr};
pub use reader::BitReader;
pub use writer::BitWriter;
