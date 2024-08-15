use std::{
    collections::HashSet,
    env,
    io::{BufRead, Write},
    mem,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use async_compression::{
    tokio::{bufread::GzipDecoder, write::GzipEncoder},
    Level,
};
use chrono::{DateTime, Utc};
use pyo3::{pyfunction, PyResult};
use rand::{rngs::StdRng, Rng, SeedableRng};

use reqwest::blocking;
use scraper::{Html, Selector};
use tar::Archive;
use tempfile::{tempdir, NamedTempFile, TempDir};

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::Command,
    runtime,
    sync::Semaphore,
};

use super::chunk_parser::{generate_q_target, transmute_slice, LeelaV6Data};

use futures::StreamExt;

const URL: &str = "https://storage.lczero.org/files/training_data/test80/";

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
    env::set_var("RUST_BACKTRACE", "1");
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

    files.reverse();

    let file_count = Arc::new(AtomicUsize::default());

    if !Path::new(output_dir).exists() {
        std::fs::create_dir_all(output_dir)?;
    }

    let mut excluded_names: HashSet<String> = HashSet::new();
    let excluded_files = std::fs::File::open(excluded_files_path);

    if let Ok(file) = excluded_files {
        let reader = std::io::BufReader::new(file);
        for item in reader.lines() {
            excluded_names.insert(item?);
        }
    }

    let builder = runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(num_threads as usize)
        .build()?;

    let mut handles = vec![];
    let processed_files = Arc::new(Mutex::new(vec![]));

    let downloader_limit = Arc::new(Semaphore::new(num_threads as usize));
    let rescorer_limit = Arc::new(Semaphore::new(num_threads as usize));

    for item in files
        .iter()
        .filter(|&&a| {
            if excluded_names.contains(a) {
                println!("Skipping {a} as it has been downloaded before");
                false
            } else {
                true
            }
        })
        .take(max_files)
    {
        let item = item.to_string();
        let output_dir = output_dir.to_string();
        let rescorer_path = rescorer_path.to_string();
        let syzygy_path = syzygy_path.to_string();
        let rescorer_args = rescorer_args.clone();

        let processed_files = Arc::clone(&processed_files);
        let file_count = Arc::clone(&file_count);
        let downloader_limit = Arc::clone(&downloader_limit);
        let rescorer_limit = Arc::clone(&rescorer_limit);
        let handle = builder.spawn(async move {
            let downloader_permit = downloader_limit.acquire().await?;
            let downloaded_dir = download_data(&item, &processed_files).await?;
            drop(downloader_permit);

            let rescorer_permit = rescorer_limit.acquire().await?;

            let rescored_files = rescore_data(
                &rescorer_path,
                downloaded_dir
                    .path()
                    .join(item.strip_suffix(".tar").unwrap())
                    .to_str()
                    .unwrap(),
                &syzygy_path,
                &rescorer_args,
                true,
            )
            .await?;

            drop(rescorer_permit);

            preprocess_data(
                tokio::fs::read_dir(rescored_files.path()).await?,
                &output_dir,
                &file_count,
                sample_rate,
                uncertainty_lambda,
                true,
            )
            .await?;

            println!("Data from {item} was successfully written to {output_dir}.");
            anyhow::Ok(())
        });

        handles.push(handle);
    }

    builder.block_on(async {
        for handle in handles {
            handle.await??;
        }
        anyhow::Ok(())
    })?;

    let mut processed_files_output = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(format!("{output_dir}/processed_tars.txt"))?,
    );

    for item in processed_files.lock().unwrap().iter() {
        writeln!(processed_files_output, "{item}")?;
    }

    Ok(())
}

async fn download_data(
    href: &str,
    processed_files: &Arc<Mutex<Vec<String>>>,
) -> anyhow::Result<TempDir> {
    let url = format!("{URL}{href}");

    println!("Downloading file {href}!");

    let downloaded_file = NamedTempFile::new()?;

    let mut downloaded_writer = std::io::BufWriter::with_capacity(1 << 20, &downloaded_file);

    let mut bytes_stream = reqwest::get(&url).await?.bytes_stream();

    while let Some(bytes) = bytes_stream.next().await {
        downloaded_writer.write_all(&bytes?)?;
    }

    downloaded_writer.flush()?;
    drop(downloaded_writer);

    let temp_path = downloaded_file.into_temp_path();
    let mut archive = Archive::new(std::io::BufReader::with_capacity(
        1 << 20,
        std::fs::File::open(temp_path)?,
    ));

    processed_files.lock().unwrap().push(href.to_string());

    let downloaded_dir = tempdir()?;

    archive.unpack(&downloaded_dir)?;

    println!("Successfully downloaded file {href}!");

    Ok(downloaded_dir)
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
) -> anyhow::Result<()> {
    env::set_var("RUST_BACKTRACE", "1");

    if !Path::new(output_dir).exists() {
        std::fs::create_dir_all(output_dir)?;
    }

    runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            if !rescorer_path.is_empty() {
                let rescored_files = rescore_data(
                    rescorer_path,
                    input_dir,
                    syzygy_path,
                    &rescorer_args,
                    delete_original,
                )
                .await?;
                preprocess_data(
                    tokio::fs::read_dir(rescored_files.path()).await?,
                    output_dir,
                    &Arc::new(AtomicUsize::default()),
                    sample_rate,
                    uncertainty_lambda,
                    true,
                )
                .await
            } else {
                preprocess_data(
                    tokio::fs::read_dir(input_dir).await?,
                    output_dir,
                    &Arc::new(AtomicUsize::default()),
                    sample_rate,
                    uncertainty_lambda,
                    true,
                )
                .await
            }
        })
}

async fn rescore_data(
    rescorer_path: &str,
    input_dir: &str,
    syzygy_paths: &str,
    rescorer_args: &[String],
    delete_original: bool,
) -> Result<TempDir, std::io::Error> {
    let should_delete = if delete_original {
        "--delete-files"
    } else {
        "--no-delete-files"
    };
    let rescored_dir = tempdir()?;
    println!("Rescoring data...");
    let mut rescorer = Command::new(rescorer_path)
        .arg("rescore")
        .arg(format!("--input={input_dir}"))
        .arg(format!("--output={}", rescored_dir.path().display()))
        .arg(format!("--syzygy-paths={syzygy_paths}"))
        .arg(should_delete)
        .args(rescorer_args)
        .spawn()?;

    let result = rescorer.wait().await?;

    if !result.success() {
        return Err(std::io::Error::other("Failed to run rescorer!"));
    }

    Ok(rescored_dir)
}

async fn preprocess_data(
    mut input_files: tokio::fs::ReadDir,
    output_dir: &str,
    file_count: &Arc<AtomicUsize>,
    sample_rate: u32,
    uncertainty_lambda: f32,
    delete_original: bool,
) -> anyhow::Result<()> {
    const POS_PER_FILE: usize = 4096;

    let curr_time = Utc::now();

    let mut input_bytes_buffer = Vec::with_capacity(mem::size_of::<LeelaV6Data>() * 160);

    let mut chunk_process_buffer = Vec::with_capacity(160);

    let mut output_buffer: Vec<u8> =
        Vec::with_capacity(POS_PER_FILE * mem::size_of::<LeelaV6Data>());
    let mut num_items = 0;

    let mut skip_rng = StdRng::from_entropy();

    while let Some(file) = input_files.next_entry().await? {
        let path = file.path();
        if path.ends_with("LICENSE") {
            continue;
        }

        let file = tokio::fs::File::open(&path).await?;

        let mut reader = GzipDecoder::new(tokio::io::BufReader::new(file));
        input_bytes_buffer.clear();
        if reader.read_to_end(&mut input_bytes_buffer).await.is_err() {
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
                    write_data(&output_buffer, output_dir, file_count, &curr_time).await?;
                    output_buffer.clear();
                    num_items = 0;
                }
            }
        }

        if delete_original {
            tokio::fs::remove_file(path).await?;
        }
    }

    if !output_buffer.is_empty() {
        write_data(&output_buffer, output_dir, file_count, &curr_time).await?
    }

    Ok(())
}

async fn write_data(
    output_buffer: &[u8],
    output_dir: &str,
    file_count: &Arc<AtomicUsize>,
    time: &DateTime<Utc>,
) -> anyhow::Result<()> {
    let new_file = tokio::fs::File::create_new(format!(
        "{output_dir}/training_processed_{}_{}.gz",
        time.format("%Y%m%d-%H%M"),
        file_count.fetch_add(1, Ordering::Relaxed)
    ))
    .await?;

    let mut gz_encoder = GzipEncoder::with_quality(new_file, Level::Best);

    gz_encoder.write_all(output_buffer).await?;
    gz_encoder.flush().await?;
    gz_encoder.shutdown().await?;
    Ok(())
}
