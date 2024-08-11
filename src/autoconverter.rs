use std::{
    fs::{self, File, ReadDir},
    io::{BufReader, BufWriter, Read, Write},
    mem,
    path::Path,
    process::Command,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use chrono::{DateTime, Utc};
use flate2::{bufread::GzDecoder, write::GzEncoder, Compression};
use pyo3::{pyfunction, PyResult};
use rand::Rng;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    ThreadPoolBuilder,
};
use reqwest::blocking;
use scraper::{Html, Selector};
use tar::Archive;
use tempfile::tempdir;

use super::chunk_parser::{generate_q_target, transmute_slice, LeelaV6Data};



#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn auto_convert(
    output_dir: &str,
    rescorer_path: &str,
    syzygy_path: &str,
    extra_args: &str,
    max_files: usize,
    num_threads: u32,
    sample_rate: u32,
    uncertainty_lambda: f32,
) -> PyResult<()> {
    const URL: &str = "https://storage.lczero.org/files/training_data/test80/";
    let response = blocking::get(URL).unwrap().text().unwrap();
    let document = Html::parse_document(&response);
    let selector = Selector::parse("a").unwrap();

    let mut files = Vec::new();

    for element in document.select(&selector) {
        if let Some(href) = element.value().attr("href") {
            if href.ends_with(".tar") {
                files.push(href);
            }
        }
    }

    let threadpool = ThreadPoolBuilder::new()
        .num_threads(num_threads as usize)
        .build()
        .unwrap();

    files.reverse();

    let file_count = Arc::new(AtomicUsize::default());

    if !Path::new(output_dir).exists() {
        fs::create_dir_all(output_dir)?;
    }

    let scratch_dir = tempdir()?;

    threadpool.install(|| {
        files
            .par_iter()
            .take(max_files)
            .try_for_each(|href| -> PyResult<()> {
                let url = format!("{URL}{href}");

                let chunks = href.trim_end_matches(".tar");

                let folder_path = scratch_dir.path().join(chunks);

                let mut response = Archive::new(BufReader::with_capacity(
                    1 << 30,
                    blocking::get(url).unwrap(),
                ));

                response.unpack(folder_path)?;

                let rescored_dir = scratch_dir.path().join(format!("rescored_{chunks}"));

                let mut rescorer = Command::new(rescorer_path)
                    .arg(format!("--input={chunks}"))
                    .arg(format!("--output={}", rescored_dir.display()))
                    .arg(format!("--syzygy-paths={syzygy_path}"))
                    .args(extra_args.split_whitespace())
                    .spawn()?;
                rescorer.wait()?;

                let rescored_files = fs::read_dir(&rescored_dir)?;

                preprocess_data(
                    rescored_files,
                    output_dir,
                    &file_count,
                    sample_rate,
                    uncertainty_lambda,
                    true,
                )?;
                Ok(())
            })
    })?;

    Ok(())
}

#[pyfunction]
pub fn process_leela_data(
    input_dir: &str,
    output_dir: &str,
    sample_rate: u32,
    uncertainty_lambda: f32,
    delete_original: bool,
) -> PyResult<()> {
    let dir = fs::read_dir(input_dir)?;
    preprocess_data(
        dir,
        output_dir,
        &Arc::new(AtomicUsize::default()),
        sample_rate,
        uncertainty_lambda,
        delete_original,
    )
}

fn preprocess_data(
    input_files: ReadDir,
    output_dir: &str,
    file_count: &Arc<AtomicUsize>,
    sample_rate: u32,
    uncertainty_lambda: f32,
    delete_original: bool,
) -> PyResult<()> {
    const POS_PER_FILE: usize = 4096;

    let curr_time = Utc::now();

    let mut input_bytes_buffer = Vec::with_capacity(mem::size_of::<LeelaV6Data>() * 160);

    let mut chunk_process_buffer = Vec::with_capacity(160);

    let mut output_buffer: Vec<u8> =
        Vec::with_capacity(POS_PER_FILE * mem::size_of::<LeelaV6Data>());
    let mut num_items = 0;

    let mut skip_rng = rand::thread_rng();

    for file in input_files {
        let path = file?.path();
        let file = File::open(&path)?;

        let mut reader = GzDecoder::new(BufReader::new(file));

        input_bytes_buffer.clear();
        reader.read_to_end(&mut input_bytes_buffer)?;

        let leela_chunk: &[LeelaV6Data] = unsafe { transmute_slice(&input_bytes_buffer) };

        chunk_process_buffer.clear();
        chunk_process_buffer.extend_from_slice(leela_chunk);
        generate_q_target(&mut chunk_process_buffer, uncertainty_lambda, 0.1);

        for item in chunk_process_buffer.iter() {
            // skip most positions to avoid sampling from the same game too many times
            if skip_rng.gen_ratio(1, sample_rate) {
                let bytes: [u8; mem::size_of::<LeelaV6Data>()] =
                    unsafe { mem::transmute_copy(item) };
                output_buffer.extend_from_slice(&bytes);
                num_items += 1;

                if num_items >= POS_PER_FILE {
                    write_data(&output_buffer, output_dir, file_count, &curr_time)?;
                    output_buffer.clear();
                    num_items = 0;
                }
            }
        }

        if delete_original {
            fs::remove_file(path)?;
        }
    }

    if !output_buffer.is_empty() {
        write_data(&output_buffer, output_dir, file_count, &curr_time)?
    }

    Ok(())
}

fn write_data(
    output_buffer: &[u8],
    output_dir: &str,
    file_count: &Arc<AtomicUsize>,
    time: &DateTime<Utc>,
) -> PyResult<()> {
    let new_file = File::create_new(format!(
        "{}/training_processed_{}_{}.gz",
        output_dir,
        time.format("%Y%m%d-%H%M"),
        file_count.fetch_add(1, Ordering::Relaxed)
    ))?;

    let mut gz_encoder = GzEncoder::new(BufWriter::new(new_file), Compression::best());

    gz_encoder.write_all(output_buffer)?;
    gz_encoder.finish()?.flush()?;
    Ok(())
}
