use std::{
    fs, mem,
    path::Path,
    sync::{Arc, Mutex},
    thread,
};

use crossbeam::channel::{self, Receiver};
use numpy::ndarray::{Array1, Array3};
use rand::{rngs::ThreadRng, seq::SliceRandom};

use crate::data_processing::file_parser::DataWorker;

pub struct DataLoader {
    shufflebuffer: ShuffleBuffer,
    batch_receiver: Receiver<Option<BatchItem>>,
    active_thread_count: usize,
}

impl DataLoader {
    pub fn new(
        directory: impl AsRef<Path>,
        num_threads: usize,
        max_shufflebuffer_capacity: usize,
        batch_size: usize,
    ) -> Option<DataLoader> {
        let (batch_sender, batch_receiver) = channel::bounded(batch_size * num_threads);
        let mut spawned_threads = 0;
        let dir = Arc::new(Mutex::new(fs::read_dir(directory).ok()?));

        for _ in 0..num_threads {
            let new_worker = DataWorker::new(&dir, &batch_sender);
            if let Some(mut worker) = new_worker {
                thread::spawn(move || worker.process_loop());
                spawned_threads += 1;
            } else {
                break;
            }
        }

        if spawned_threads == 0 {
            return None;
        }

        let shufflebuffer = ShuffleBuffer::new_uninit(max_shufflebuffer_capacity);

        let mut dataloader = DataLoader {
            shufflebuffer,
            batch_receiver,
            active_thread_count: spawned_threads,
        };

        dataloader.fill_shufflebuffer();

        Some(dataloader)
    }

    pub fn next_item_shuffled(&mut self) -> Option<BatchItem> {
        let next_item = self.next_item_unshuffled();
        self.shufflebuffer.random_replace(next_item)
    }

    fn fill_shufflebuffer(&mut self) {
        while self.shufflebuffer.items.len() < self.shufflebuffer.items.capacity() {
            if let Some(item) = self.next_item_unshuffled() {
                self.shufflebuffer.items.push(item);
            } else {
                break;
            }
        }
        self.shufflebuffer.shuffle();
    }

    // Worker threads will continuously send batch items until they have finished processing data, at which point
    // they will send a "None" to signal that they have completed their tasks
    fn next_item_unshuffled(&mut self) -> Option<BatchItem> {
        while self.active_thread_count > 0 {
            let new_item = self.batch_receiver.recv().expect("Channel disconnected!");
            match new_item {
                item @ Some(_) => return item,
                None => {
                    self.active_thread_count -= 1;
                }
            }
        }
        None
    }
}

struct ShuffleBuffer {
    items: Vec<BatchItem>,
    rng: ThreadRng,
}

impl ShuffleBuffer {
    fn new_uninit(max_capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(max_capacity),
            rng: rand::thread_rng(),
        }
    }

    fn random_replace(&mut self, replace_item: Option<BatchItem>) -> Option<BatchItem> {
        match replace_item {
            Some(item) => {
                let to_replace = self.items.choose_mut(&mut self.rng).unwrap();
                Some(mem::replace(to_replace, item))
            }
            None => self.items.pop(),
        }
    }

    fn shuffle(&mut self) {
        self.items.shuffle(&mut self.rng)
    }
}

#[derive(Default)]
pub struct BatchItem {
    pub input_planes: Array3<f32>,
    pub policy_target: Array1<f32>,
    pub wdl_target: [f32; 3],
    pub q_target: f32,
    pub mlh_target: f32,
}
