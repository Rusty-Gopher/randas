// src/io/read.rs

use crate::dataframe::dataframe::{DataFrame, DataFrameValue};
use crate::errors::MyError;
use calamine::{open_workbook_auto, Data, Reader};
use csv::ReaderBuilder;
use csv::StringRecord;
use encoding_rs::Encoding;
use rayon::prelude::*;
use rusqlite::{Connection, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub fn read_excel<P: AsRef<Path>>(
    path: P,
    sheet_name: Option<&str>,
) -> Result<DataFrame, Box<dyn Error>> {
    let mut workbook = open_workbook_auto(path.as_ref())?;

    let sheet_name = match sheet_name {
        Some(name) => name.to_string(),
        None => workbook
            .sheet_names()
            .get(0)
            .ok_or("No sheets found in Excel file")?
            .to_string(),
    };

    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|e| format!("Failed to get range from sheet due to error: {}", e))?;

    let mut columns: Vec<String> = Vec::new();
    let mut data: HashMap<String, Vec<Option<DataFrameValue>>> = HashMap::new();
    let mut is_header = true;

    for row in range.rows() {
        if is_header {
            columns = row.iter().map(|cell| cell.to_string()).collect::<Vec<_>>();
            for column in &columns {
                data.insert(column.clone(), Vec::new());
            }
            is_header = false;
        } else {
            for (idx, cell) in row.iter().enumerate() {
                let cell_value = match cell {
                    Data::Int(i) => Some(DataFrameValue::Integer(*i)),
                    Data::Float(f) => Some(DataFrameValue::Float(*f)),
                    Data::String(s) => Some(DataFrameValue::String(s.clone())),
                    Data::Bool(b) => Some(DataFrameValue::Boolean(*b)),
                    // i will add more if its needed in future
                    _ => None,
                };

                if let Some(column_name) = columns.get(idx) {
                    data.get_mut(column_name).unwrap().push(cell_value);
                }
            }
        }
    }

    DataFrame::new(data, columns).map_err(Into::into)
}

// batch processing for read csv
fn process_batch(
    chunk: &[StringRecord],
    columns: &[String],
) -> HashMap<String, Vec<Option<DataFrameValue>>> {
    let mut batch_data: HashMap<String, Vec<Option<DataFrameValue>>> = HashMap::new();

    for record in chunk {
        for (i, field) in record.iter().enumerate() {
            let column_name = &columns[i];
            let value = infer_data_type(field);
            batch_data
                .entry(column_name.clone())
                .or_insert_with(Vec::new)
                .push(value);
        }
    }

    batch_data
}

// Function to read CSV with delimiter and encoding options
pub fn read_csv(
    file_path: &str,
    delimiter: u8,
    encoding: &'static Encoding,
) -> Result<DataFrame, Box<dyn Error>> {
    let mut file = File::open(file_path)?;
    let mut file_contents = Vec::new();
    file.read_to_end(&mut file_contents)?;

    // Decode the file content to UTF-8
    let (file_content_utf8, _, _) = encoding.decode(&file_contents);

    // Create the CSV reader with specified delimiter
    let mut rdr = ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(true)
        .from_reader(file_content_utf8.as_bytes());

    let headers = rdr.headers()?.clone();
    let columns: Vec<String> = headers.iter().map(String::from).collect();

    // Collect records for further processing
    let records: Vec<StringRecord> = rdr.records().filter_map(Result::ok).collect();

    // Process each chunk of records in parallel
    let data: HashMap<String, Vec<Option<DataFrameValue>>> = records
        .par_chunks(1000) // Adjust chunk size based on performance and memory usage
        .map(|chunk| process_batch(chunk, &columns))
        .reduce_with(|mut a, b| {
            for (column, values) in b {
                a.entry(column).or_default().extend(values);
            }
            a
        })
        .unwrap_or_default();

    DataFrame::new(data, columns).map_err(|e| e.into())
}

fn infer_data_type(field: &str) -> Option<DataFrameValue> {
    if field.is_empty() {
        None
    } else if let Ok(int_val) = field.parse::<i64>() {
        Some(DataFrameValue::Integer(int_val))
    } else if let Ok(float_val) = field.parse::<f64>() {
        Some(DataFrameValue::Float(float_val))
    } else {
        Some(DataFrameValue::String(field.to_string()))
    }
}

// A recursive function to flatten JSON objects into a series of key-value pairs.
fn flatten_json(prefix: String, value: &Value, flattened: &mut Vec<HashMap<String, Value>>) {
    match value {
        Value::Object(obj) => {
            let mut base = HashMap::new();
            for (key, value) in obj {
                // For nested objects, prefix keys with the parent key.
                let new_key = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}_{}", prefix, key)
                };
                match value {
                    Value::Array(arr) => {
                        for val in arr {
                            flatten_json(new_key.clone(), val, flattened);
                        }
                    }
                    Value::Object(_) => {
                        flatten_json(new_key, value, flattened);
                    }
                    _ => {
                        base.insert(new_key, value.clone());
                    }
                }
            }
            if !base.is_empty() {
                flattened.push(base);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                flatten_json(prefix.clone(), item, flattened);
            }
        }
        _ => {
            // insert null directly.
            let mut record = HashMap::new();
            record.insert(prefix, value.clone());
            flattened.push(record);
        }
    }
}

// Function to convert a record (Vec of HashMaps) to a DataFrame
fn record_to_dataframe(
    records: Vec<HashMap<String, Value>>,
) -> Result<DataFrame, Box<dyn std::error::Error>> {
    if records.is_empty() {
        return Err("No records to process.".into());
    }

    let mut data: HashMap<String, Vec<Option<DataFrameValue>>> = HashMap::new();

    // Add the DataFrame columns from each record
    for record in records.iter() {
        for (column_name, value) in record.iter() {
            let dataframe_value = match value {
                Value::String(s) => Some(DataFrameValue::String(s.clone())),
                Value::Number(n) if n.is_f64() => Some(DataFrameValue::Float(n.as_f64().unwrap())),
                Value::Number(n) if n.is_i64() => {
                    Some(DataFrameValue::Integer(n.as_i64().unwrap()))
                }
                Value::Bool(b) => Some(DataFrameValue::Boolean(*b)),
                _ => None,
            };

            data.entry(column_name.clone())
                .or_insert_with(Vec::new)
                .push(dataframe_value);
        }
    }

    // Ensure all columns have the same length by extending shorter ones with None
    let max_length = data.values().map(|v| v.len()).max().unwrap_or(0);
    for column_values in data.values_mut() {
        while column_values.len() < max_length {
            column_values.push(None);
        }
    }

    // Initialize columns after determining which columns are present
    let columns = data.keys().cloned().collect::<Vec<_>>();

    // Verify column lengths one more time (optional, for sanity check)
    if data.values().any(|v| v.len() != max_length) {
        return Err("Column length normalization failed.".into());
    }

    DataFrame::new(data, columns).map_err(Into::into)
}

// Main function to read a JSON and convert it to a DataFrame
pub fn read_json_to_dataframe(file_path: &str) -> Result<DataFrame, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let json: Value = serde_json::from_reader(file)?;

    let mut records = Vec::new();
    flatten_json("".to_string(), &json, &mut records);

    let dataframe = record_to_dataframe(records)?;
    Ok(dataframe)
}

pub fn read_sql(conn: &Connection, query: &str) -> Result<DataFrame, MyError> {
    let mut stmt = conn.prepare(query)?;

    let columns: Vec<String> = stmt
        .column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    // Initialize a HashMap to store column data.
    let mut data: HashMap<String, Vec<Option<DataFrameValue>>> = HashMap::new();
    for column in &columns {
        data.insert(column.clone(), Vec::new());
    }

    // Use `query_map` to iterate over rows in the result set.
    let rows = stmt.query_map([], |row| {
        let mut row_data = HashMap::new();
        for (idx, column_name) in columns.iter().enumerate() {
            // Lets handle the conversion of SQLite data types to our DataFrameValue.
            let value = match row.get_ref_unwrap(idx).data_type() {
                rusqlite::types::Type::Integer => Some(DataFrameValue::Integer(row.get(idx)?)),
                rusqlite::types::Type::Real => Some(DataFrameValue::Float(row.get(idx)?)),
                rusqlite::types::Type::Text => Some(DataFrameValue::String(row.get(idx)?)),
                _ => None, // Lets add it later, when its needed but for now its enough
            };
            row_data.insert(column_name.to_string(), value);
        }
        Ok(row_data)
    })?;

    // Iterate over each row result, populating the `data` HashMap.
    for row_result in rows {
        let row_data = row_result?;
        for (column_name, value) in row_data.iter() {
            if let Some(column_values) = data.get_mut(column_name) {
                column_values.push(value.clone());
            }
        }
    }

    // create a dataframe
    DataFrame::new(data, columns)
        .map_err(|e| MyError::DataFrame(format!("Failed to create DataFrame: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use encoding_rs::UTF_8;
    use rusqlite::Connection;

    #[test]
    fn test_read_excel() {
        // This is coming from examples folder, which is added for unit testing
        let test_excel_path = "examples\\data.xlsx";

        let result = read_excel(test_excel_path, None);

        // Check if the result is Ok and unwrap the DataFrame
        match result {
            Ok(dataframe) => {
                let (rows, columns) = dataframe.shape();

                assert_eq!(
                    rows, 4,
                    "The number of rows does not match the expected value."
                );
                assert_eq!(
                    columns, 4,
                    "The number of columns does not match the expected value."
                );
            }
            Err(e) => {
                // Panicccc, with a clear msg that the file isnt opening
                panic!("Failed to read Excel file: {:?}", e);
            }
        }
    }

    #[test]
    fn test_parallel_csv_reading() {
        // This is coming from examples folder, which is added for unit testing
        let file_path = "examples\\sample.csv";

        let result = read_csv(file_path, b',', UTF_8);

        // Check that the result is Ok and contains the expected data
        assert!(result.is_ok());
        let dataframe = result.unwrap();

        // Verify the columns
        assert_eq!(dataframe.columns, vec!["Name", "Age", "Gender"]);
        assert_eq!(dataframe.data.get("Name").unwrap().len(), 4);
        assert_eq!(dataframe.data.get("Age").unwrap().len(), 4);
        assert_eq!(dataframe.data.get("Gender").unwrap().len(), 4);
    }

    #[test]
    fn test_read_json() {
        let test_json_path = "examples\\sample.json";

        // Call the read_json function
        let dataframe = read_json_to_dataframe(test_json_path);

        match dataframe {
            Ok(dataframe) => {
                // Verify specific data in the DataFrame to ensure it read correctly
                if let Some(column_data) = dataframe.column("firstName") {
                    assert_eq!(
                        column_data.get(0).unwrap(),
                        &Some(DataFrameValue::String("Joe".to_string())),
                        "First name should be Joe"
                    );
                } else {
                    panic!("Column 'firstName' not found");
                }
            }
            Err(e) => {
                // Handle the error case
                panic!("Failed to read DataFrame: {:?}", e);
            }
        }
    }

    // Function to set up a test database (in-memory for testing purposes)
    fn setup_test_db() -> rusqlite::Result<Connection> {
        let conn = Connection::open_in_memory()?;
        conn.execute(
            "CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER)",
            [],
        )?;
        conn.execute(
            "INSERT INTO people (name, age) VALUES (?1, ?2)",
            &["Alice", "32"],
        )?;
        conn.execute(
            "INSERT INTO people (name, age) VALUES (?1, ?2)",
            &["Bob", "45"],
        )?;
        Ok(conn)
    }
    #[test]
    fn test_read_sql() {
        let conn = setup_test_db().unwrap();

        let query = "SELECT * FROM people";
        let dataframe = read_sql(&conn, query).unwrap();

        // 1. Check for DataFrame shape
        let expected_rows = 2;
        let expected_cols = 3;
        assert_eq!(
            dataframe.shape(),
            (expected_rows, expected_cols),
            "DataFrame shape does not match"
        );

        // 2. Check for column names
        let expected_columns = vec!["id", "name", "age"];
        assert_eq!(
            dataframe.columns, expected_columns,
            "DataFrame columns do not match"
        );

        // 3. Checking the first row's data
        assert_eq!(
            dataframe.get(0, "id"),
            Some(&Some(DataFrameValue::Integer(1))),
            "ID of the first row does not match"
        );
        assert_eq!(
            dataframe.get(0, "name"),
            Some(&Some(DataFrameValue::String("Alice".to_string()))),
            "Name of the first row does not match"
        );
        assert_eq!(
            dataframe.get(0, "age"),
            Some(&Some(DataFrameValue::Integer(32))),
            "Age of the first row does not match"
        );
    }
}
