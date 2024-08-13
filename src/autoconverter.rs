use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions, ReadDir},
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    mem,
    path::Path,
    process::Command,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use chrono::{DateTime, Utc};
use flate2::{bufread::GzDecoder, Compression, GzBuilder};
use pyo3::{pyfunction, PyResult};
use rand::Rng;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    ThreadPoolBuilder,
};
use reqwest::blocking;
use scraper::{Html, Selector};
use tar::Archive;
use tempfile::{tempdir, TempDir};

use super::chunk_parser::{generate_q_target, transmute_slice, LeelaV6Data};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn auto_convert(
    output_dir: &str,
    rescorer_path: &str,
    syzygy_path: &str,
    rescorer_args: Vec<String>,
    max_files: usize,
    num_threads: u32,
    sample_rate: u32,
    uncertainty_lambda: f32,
    excluded_files_path: &str,
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

    let mut excluded_names: HashSet<String> = HashSet::new();
    let excluded_files = File::open(excluded_files_path);

    if let Ok(file) = excluded_files {
        let reader = BufReader::new(file);
        for item in reader.lines() {
            excluded_names.insert(item?);
        }
    }

    let processed_files = Arc::new(Mutex::new(vec![]));

    threadpool.install(|| {
        files
            .par_iter()
            .take(max_files)
            .filter(|&&a| !excluded_names.contains(a))
            .try_for_each(|href| -> PyResult<()> {
                let url = format!("{URL}{href}");

                println!("Downloading file {href}!");

                let mut response = Archive::new(BufReader::with_capacity(
                    1 << 30,
                    blocking::get(url).unwrap(),
                ));

                println!("Successfully downloaded file {href}!");

                processed_files.lock().unwrap().push(href.to_string());

                let downloaded_dir = tempdir()?;

                response.unpack(&downloaded_dir)?;

                let rescored_files = rescore_data(
                    rescorer_path,
                    downloaded_dir
                        .path()
                        .join(href.strip_suffix(".tar").unwrap())
                        .to_str()
                        .unwrap(),
                    syzygy_path,
                    &rescorer_args,
                    true,
                )?;


                preprocess_data(
                    fs::read_dir(rescored_files.path())?,
                    output_dir,
                    &file_count,
                    sample_rate,
                    uncertainty_lambda,
                    true,
                )?;

                println!("Data from {href} was successfully written to {output_dir}.");

                Ok(())
            })
    })?;

    let mut processed_files_output = BufWriter::new(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(format!("{output_dir}/processed_tars.txt"))?,
    );

    for item in processed_files.lock().unwrap().iter() {
        writeln!(processed_files_output, "{item}")?;
    }

    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn process_leela_data(
    input_dir: &str,
    output_dir: &str,
    rescorer_path: &str,
    syzygy_path: &str,
    rescorer_args: Vec<String>,
    sample_rate: u32,
    uncertainty_lambda: f32,
    delete_original: bool,
) -> PyResult<()> {
    if !Path::new(output_dir).exists() {
        fs::create_dir_all(output_dir)?;
    }
    if !rescorer_path.is_empty() {
        println!("Rescoring file {input_dir}...");
        let rescored_files = rescore_data(
            rescorer_path,
            input_dir,
            syzygy_path,
            &rescorer_args,
            delete_original,
        )?;
        preprocess_data(
            fs::read_dir(rescored_files.path())?,
            output_dir,
            &Arc::new(AtomicUsize::default()),
            sample_rate,
            uncertainty_lambda,
            true,
        )
    } else {
        preprocess_data(
            fs::read_dir(input_dir)?,
            output_dir,
            &Arc::new(AtomicUsize::default()),
            sample_rate,
            uncertainty_lambda,
            delete_original,
        )
    }
}

fn rescore_data(
    rescorer_path: &str,
    input_dir: &str,
    syzygy_paths: &str,
    rescorer_args: &[String],
    delete_original: bool,
) -> Result<TempDir, io::Error> {
    let should_delete = if delete_original {
        "--delete-files"
    } else {
        "--no-delete-files"
    };
    let rescored_dir = tempdir()?;
    let mut rescorer = Command::new(rescorer_path)
        .arg("rescore")
        .arg(format!("--input={input_dir}"))
        .arg(format!("--output={}", rescored_dir.path().display()))
        .arg(format!("--syzygy-paths={syzygy_paths}"))
        .arg(should_delete)
        .args(rescorer_args)
        .spawn()?;
    let result = rescorer.wait()?;
    if !result.success() {
        return Err(io::Error::other("Failed to run rescorer!"));
    }

    Ok(rescored_dir)
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
        if path.ends_with("LICENSE") {
            continue;
        }

        let file = File::open(&path)?;

        let mut reader = GzDecoder::new(BufReader::new(file));
        input_bytes_buffer.clear();
        if reader.read_to_end(&mut input_bytes_buffer).is_err() {
            continue;
        }

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
        "{output_dir}/training_processed_{}_{}.gz",
        time.format("%Y%m%d-%H%M"),
        file_count.fetch_add(1, Ordering::Relaxed)
    ))?;

    let mut gz_encoder = GzBuilder::new().write(new_file, Compression::best());

    gz_encoder.write_all(output_buffer)?;
    gz_encoder.finish()?.flush()?;
    Ok(())
}
