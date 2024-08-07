use std::{
    fs::{File, ReadDir},
    io::{BufReader, Read},
    sync::{Arc, Mutex},
};

use crossbeam::channel::Sender;
use flate2::bufread::GzDecoder;

use super::{chunk_parser::ChunkParser, dataloader::BatchItem};

pub const CHUNK_CAPACITY: usize = 1 << 25; // 32 MB

pub struct DataWorker {
    directory: Arc<Mutex<ReadDir>>,
    chunk_parser: ChunkParser,
    batch_sender: Sender<Option<BatchItem>>,
}

impl DataWorker {
    pub fn new(
        directory: &Arc<Mutex<ReadDir>>,
        sender_queue: &Sender<Option<BatchItem>>,
    ) -> Option<Self> {
    
        let chunk_parser = ChunkParser::default();
        Some(Self {
            directory: Arc::clone(directory),
            chunk_parser,
            batch_sender: sender_queue.clone(),
        })
    }

    pub fn process_loop(&mut self) {
        loop {
            let next = self.next_item();
            let should_stop = next.is_none();
            self.batch_sender.send(next).unwrap();
            if should_stop {
                break;
            }
        }
    }

    fn next_item(&mut self) -> Option<BatchItem> {
        let next_item = self.chunk_parser.next_item();
        match next_item {
            item @ Some(_) => item,
            None => {
                self.chunk_parser.clear();
                self.fill_next_chunk()?;
                self.next_item()
            }
        }
    }

    fn fill_next_chunk(&mut self) -> Option<()> {
        let next_chunk = self.directory.lock().unwrap().next()?.unwrap();
        let file = File::open(next_chunk.path()).unwrap();
        let mut chunk = GzDecoder::new(BufReader::with_capacity(CHUNK_CAPACITY, file));
        chunk.read_to_end(&mut self.chunk_parser.buffer).unwrap();
        Some(())
    }


}
