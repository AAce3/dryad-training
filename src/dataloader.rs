use std::{
    fs, mem,
    sync::{Arc, Mutex},
    thread,
};

use crossbeam::channel::{self, Receiver};
use pyo3::{exceptions::PyValueError, pyclass, pymethods, PyErr, PyResult};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use super::file_parser::DataWorker;

#[pyclass]
pub struct DataLoader {
    shufflebuffer: ShuffleBuffer,
    batch_receiver: Receiver<Option<BatchItem>>,
    active_thread_count: usize,
}

#[pymethods]
impl DataLoader {
    #[new]
    pub fn new(
        directory: &str,
        num_threads: usize,
        max_shufflebuffer_capacity: usize,
        batch_size: usize,
        diff_focus_min: f32,
        diff_focus_slope: f32,
    ) -> PyResult<Self> {
        let (batch_sender, batch_receiver) = channel::bounded(batch_size * num_threads);
        let mut spawned_threads = 0;
        let dir = Arc::new(Mutex::new(fs::read_dir(directory)?));

        for _ in 0..num_threads {
            let new_worker = DataWorker::new(&dir, &batch_sender, diff_focus_min, diff_focus_slope);
            if let Some(mut worker) = new_worker {
                thread::spawn(move || worker.process_loop());
                spawned_threads += 1;
            } else {
                break;
            }
        }

        if spawned_threads == 0 {
            return Err(PyErr::new::<PyValueError, _>(
                "Failed to create dataloader from {directory}!",
            ));
        }

        let shufflebuffer = ShuffleBuffer::new_uninit(max_shufflebuffer_capacity);

        let mut dataloader = DataLoader {
            shufflebuffer,
            batch_receiver,
            active_thread_count: spawned_threads,
        };

        dataloader.fill_shufflebuffer();

        Ok(dataloader)
    }

    pub fn next_item_shuffled(&mut self) -> Option<BatchItem> {
        let next_item = self.next_item_unshuffled();
        self.shufflebuffer.random_replace(next_item)
    }
}

impl DataLoader {
    

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
    rng: StdRng,
}

impl ShuffleBuffer {
    fn new_uninit(max_capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(max_capacity),
            rng: StdRng::from_entropy(),
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

#[pyclass]
pub struct BatchItem {
    #[pyo3(get)]
    pub input_planes: Vec<f32>,
    #[pyo3(get)]
    pub policy_target: Vec<f32>,
    #[pyo3(get)]
    pub wdl_target: [f32; 3],
    #[pyo3(get)]
    pub q_target: f32,
    #[pyo3(get)]
    pub mlh_target: f32,
}

impl Default for BatchItem {
    fn default() -> Self {
        Self {
            input_planes: vec![0.0; 112 * 8 * 8],
            policy_target: vec![0.0; 1858],
            wdl_target: Default::default(),
            q_target: Default::default(),
            mlh_target: Default::default(),
        }
    }
}
