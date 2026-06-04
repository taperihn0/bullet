/*

use std::{fs::File, sync::mpsc, thread};
use crate::game::formats::bulletformat::ChessBoard;
use super::{DataLoader, rng::SimpleRand};
use std::io::{Read, Seek};

#[derive(Clone)]
pub struct LeafPackLoader {
    file_paths: Vec<String>,
    buffer_size: usize,
    threads: usize,
}

/// A single training data entry of Leaf.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LeafTrainingDataEntry {
    pub occ: u64,
    pub nibbles: [u8; 16],
    pub score: i16,
    pub result: u8,
    pub ksq: u8,
    pub opp_ksq: u8,
    pub extra: [u8; 3],
}

fn convert_to_chessboard(entry: LeafTrainingDataEntry) -> ChessBoard {
    
}

unsafe impl bytemuck::Zeroable for LeafTrainingDataEntry {}
unsafe impl bytemuck::Pod for LeafTrainingDataEntry {}

#[derive(Debug)]
struct LeafTrainingDataEntryReader<T: Read + Seek> {
    input_file: T,
}

impl<T: Read + Seek> LeafTrainingDataEntryReader<T> {
    pub fn new(file: T) -> Option<Self> {
        let reader = Self {
            input_file: file,
        };

        Some(reader)
    }

    pub fn next(&mut self) -> Option<LeafTrainingDataEntry> {
        let num_bytes = std::mem::size_of::<LeafTrainingDataEntry>();
        let mut buffer = vec![0u8; num_bytes];

        match self.input_file.read_exact(&mut buffer) {
            Ok(()) => {
                let entry = bytemuck::cast_slice::<u8, LeafTrainingDataEntry>(&buffer)[0];
                Some(entry)
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    None
                } else {
                    panic!("Error while reading from file: {}", e);
                }
            }
        }
    }
}

impl LeafPackLoader {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, threads)
    }

    pub fn new_concat_multiple(
        paths: &[&str],
        buffer_size_mb: usize,
        threads: usize
    ) -> Self {
        Self {
            file_paths: paths.iter().map(|x| x.to_string()).collect(),
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<ChessBoard>() / 2,
            threads,
        }
    }
}

impl DataLoader<ChessBoard> for LeafPackLoader 
{
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let file_paths = self.file_paths.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;

        let reader_buffer_size = 16384 * threads;
        let (reader_sender, reader_receiver) = mpsc::sync_channel::<Vec<LeafTrainingDataEntry>>(8);
        let (reader_msg_sender, reader_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut buffer = Vec::with_capacity(reader_buffer_size);

            'dataloading: loop {
                for file in &file_paths {
                    let file = File::open(file).unwrap();
                    let mut reader = LeafTrainingDataEntryReader::new(file).unwrap();

                    while let Some(entry) = reader.next() {
                        buffer.push(entry);

                        if buffer.len() == reader_buffer_size {
                            let ready_buffer = std::mem::take(&mut buffer);

                            if reader_msg_receiver.try_recv().unwrap_or(false) || reader_sender.send(ready_buffer).is_err() {
                                break 'dataloading;
                            }

                            buffer.reserve_exact(reader_buffer_size);
                        }
                    }

                    if !buffer.is_empty() {
                        let ready_buffer = std::mem::take(&mut buffer);
                        if reader_msg_receiver.try_recv().unwrap_or(false) || reader_sender.send(ready_buffer).is_err() {
                            break 'dataloading;
                        }
                    }
                }
            }
        });

        let (converted_sender, converted_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(4 * threads);
        let (converted_msg_sender, converted_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut should_break = false;
            'dataloading: while let Ok(unfiltered) = reader_receiver.recv() {
                if should_break || converted_msg_receiver.try_recv().unwrap_or(false) {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                thread::scope(|s| {
                    let chunk_size = unfiltered.len().div_ceil(threads);
                    let mut handles = Vec::new();

                    for chunk in unfiltered.chunks(chunk_size) {
                        let this_sender = converted_sender.clone();
                        let handle = s.spawn(move || {
                            let mut buffer = Vec::with_capacity(chunk_size);

                            for entry in chunk {
                                buffer.push(convert_to_chessboard(entry));
                            }

                            this_sender.send(buffer).is_err()
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        if handle.join().unwrap() {
                            should_break = true;
                        }
                    }
                });

                if should_break {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }
            }
        });
    }
}

fn shuffle(data: &mut [ChessBoard]) {
    let mut rng = SimpleRand::with_seed();

    for i in (0..data.len()).rev() {
        let idx = rng.rng() as usize % (i + 1);
        data.swap(idx, i);
    }
}
*/
