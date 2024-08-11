pub mod autoconverter;
pub mod chunk_parser;
pub mod dataloader;
pub mod file_parser;
mod policy_names;

use autoconverter::{auto_convert, process_leela_data};

use dataloader::{BatchItem, DataLoader};
use pyo3::prelude::*;

#[pymodule]
fn data_processing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(auto_convert, m)?)?;
    m.add_function(wrap_pyfunction!(process_leela_data, m)?)?;
    m.add_class::<BatchItem>()?;
    m.add_class::<DataLoader>()?;
    Ok(())
}

