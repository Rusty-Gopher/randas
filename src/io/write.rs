// io/write.rs
use crate::dataframe::dataframe::{DataFrame, DataFrameValue};
use rusqlite::{Connection, ToSql};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use xlsxwriter::*;

pub fn to_excel(
    df: &DataFrame,
    file_path: &str,
    sheet_name: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let workbook = Workbook::new(file_path)?;
    let mut worksheet = workbook.add_worksheet(sheet_name)?;

    // Write column headers
    for (i, column_name) in df.columns.iter().enumerate() {
        worksheet.write_string(0, i as u16, column_name, None)?;
    }

    // Iterate over columns to write data
    for (col_idx, column_name) in df.columns.iter().enumerate() {
        if let Some(column_data) = df.data.get(column_name) {
            for (row_idx, value) in column_data.iter().enumerate() {
                match value {
                    Some(DataFrameValue::String(val)) => {
                        worksheet.write_string((row_idx + 1) as u32, col_idx as u16, val, None)?;
                    }
                    Some(DataFrameValue::Float(val)) => {
                        worksheet.write_number((row_idx + 1) as u32, col_idx as u16, *val, None)?;
                    }
                    Some(DataFrameValue::Integer(val)) => {
                        worksheet.write_number(
                            (row_idx + 1) as u32,
                            col_idx as u16,
                            *val as f64,
                            None,
                        )?;
                    }
                    Some(DataFrameValue::Boolean(val)) => {
                        let bool_str = if *val { "TRUE" } else { "FALSE" };
                        worksheet.write_string(
                            (row_idx + 1) as u32,
                            col_idx as u16,
                            bool_str,
                            None,
                        )?;
                    }
                    Some(DataFrameValue::DateTime(val)) => {
                        let date_str = val.format("%Y-%m-%d %H:%M:%S").to_string();
                        worksheet.write_string(
                            (row_idx + 1) as u32,
                            col_idx as u16,
                            &date_str,
                            None,
                        )?;
                    }
                    None => {
                        worksheet.write_blank((row_idx + 1) as u32, col_idx as u16, None)?;
                    }
                }
            }
        }
    }

    workbook.close().map_err(Into::into)
}

/// This function will return an error if the file cannot be opened for writing or if writing to the file fails.
pub fn to_json(dataframe: &DataFrame, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Use serde_json::to_string_pretty for human-readable formatting
    let serialized = serde_json::to_string_pretty(&dataframe)?;

    let path = Path::new(file_path);
    let mut file = File::create(&path)?;

    // Write the pretty-printed JSON to file
    file.write_all(serialized.as_bytes())?;

    Ok(())
}

// This function takes a reference to a DataFrame and writes its content to a SQL table
pub fn to_sql(
    dataframe: &DataFrame,
    table_name: &str,
    conn: &Connection,
    if_exists: &str,
) -> Result<(), Box<dyn Error>> {
    let placeholders = "?,"
        .repeat(dataframe.columns.len())
        .trim_end_matches(',')
        .to_string();

    if if_exists == "replace" {
        conn.execute(&format!("DELETE FROM {}", table_name), [])?;
    }

    // Prepare the SQL statement outside the loop
    let column_names = dataframe.columns.join(", ");
    let insert_sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        table_name, column_names, placeholders
    );
    let mut statement = conn.prepare(&insert_sql)?;

    for row_index in 0..dataframe.get_index().len() {
        let row_values: Vec<Box<dyn ToSql>> = dataframe
            .columns
            .iter()
            .map(|col| {
                match dataframe
                    .data
                    .get(col)
                    .and_then(|v| v.get(row_index).cloned().unwrap_or(None))
                {
                    Some(DataFrameValue::Integer(i)) => Box::new(i) as Box<dyn ToSql>,
                    Some(DataFrameValue::Float(f)) => Box::new(f) as Box<dyn ToSql>,
                    Some(DataFrameValue::String(s)) => Box::new(s.clone()) as Box<dyn ToSql>,
                    _ => Box::new(rusqlite::types::Value::Null),
                }
            })
            .collect();

        let params: Vec<&dyn ToSql> = row_values.iter().map(|b| b.as_ref()).collect();
        statement.execute(params.as_slice())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::dataframe::{DataFrame, DataFrameValue};
    use crate::io::read::read_excel;
    use rusqlite::Connection;
    use std::collections::HashMap;
    use std::fs;
    use std::fs::File;
    use std::io::Read;

    /// this is the sample dataframe, that i can use throughout the unit testcases
    /// Index	ID	    Name	    Score
    /// 0	    1	    Alice	    3.5
    /// 1	    2	    Bob	        4.0
    /// 2	    3	    Charlie	    2.5
    fn setup_test_dataframe() -> DataFrame {
        // Define the column names
        let columns = vec!["ID".to_string(), "Name".to_string(), "Score".to_string()];

        // Organize data by columns
        let mut data: HashMap<String, Vec<Option<DataFrameValue>>> = HashMap::new();
        data.insert(
            "ID".to_string(),
            vec![
                Some(DataFrameValue::Integer(1)),
                Some(DataFrameValue::Integer(2)),
                Some(DataFrameValue::Integer(3)),
            ],
        );
        data.insert(
            "Name".to_string(),
            vec![
                Some(DataFrameValue::String("Alice".to_string())),
                Some(DataFrameValue::String("Bob".to_string())),
                Some(DataFrameValue::String("Charlie".to_string())),
            ],
        );
        data.insert(
            "Score".to_string(),
            vec![
                Some(DataFrameValue::Float(3.5)),
                Some(DataFrameValue::Float(4.0)),
                Some(DataFrameValue::Float(2.5)),
            ],
        );

        DataFrame::new(data, columns).expect("Failed to create DataFrame")
    }

    #[test]
    fn test_to_excel_invalid_path() {
        let df = setup_test_dataframe();

        // Use an invalid path
        let file_path = "/invalid/path/test_output.xlsx";
        let result = to_excel(&df, file_path, None);

        // Verify: The function should return an error
        assert!(result.is_err());
    }

    #[test]
    fn test_excel_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        // Step 1: Create a known DataFrame
        let df = setup_test_dataframe();

        // Step 2: Write the DataFrame to an Excel file
        let file_path = "temp_test_excel_round_trip.xlsx";
        to_excel(&df, file_path, None)?;

        // Step 3: Read the Excel file back into a new DataFrame
        let df_read = read_excel(file_path, None)?;

        assert_eq!(df_read.shape(), (3, 3));

        // Cleanup: Remove the temporary Excel file
        fs::remove_file(file_path)?;

        Ok(())
    }

    #[test]
    fn test_to_json() -> Result<(), Box<dyn std::error::Error>> {
        let df = setup_test_dataframe();
        let file_path = "test_dataframe.json";

        to_json(&df, &file_path)?;

        let mut file = File::open(file_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        assert!(contents.contains("Alice"));
        assert!(contents.contains("3.5"));

        fs::remove_file(file_path)?;
        Ok(())
    }

    // Setup an in-memory SQLite database for testing
    fn setup_test_db() -> rusqlite::Result<Connection> {
        let conn = Connection::open_in_memory()?;
        conn.execute(
            "CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER)",
            [],
        )?;
        Ok(conn)
    }

    fn create_sql_dataframe() -> DataFrame {
        // Define the column names
        let columns = vec!["id".to_string(), "name".to_string(), "age".to_string()];

        // Organize data by columns
        let mut data: HashMap<String, Vec<Option<DataFrameValue>>> = HashMap::new();
        data.insert(
            "id".to_string(),
            vec![
                Some(DataFrameValue::Integer(1)),
                Some(DataFrameValue::Integer(2)),
            ],
        );
        data.insert(
            "name".to_string(),
            vec![
                Some(DataFrameValue::String("Alice".to_string())),
                Some(DataFrameValue::String("Bob".to_string())),
            ],
        );
        data.insert(
            "age".to_string(),
            vec![
                Some(DataFrameValue::Integer(32)),
                Some(DataFrameValue::Integer(45)),
            ],
        );

        // Construct the DataFrame
        let dataframe = DataFrame::new(data, columns).expect("Failed to create DataFrame");

        dataframe
    }

    #[test]
    fn test_to_sql() {
        // Set up the in-memory database and create the test DataFrame
        let conn = setup_test_db().expect("Failed to set up test database");
        let dataframe = create_sql_dataframe();

        // Use the same connection object for the `to_sql` function
        to_sql(&dataframe, "people", &conn, "replace").expect("Failed to insert DataFrame");

        // Query the database to retrieve the inserted data
        let mut stmt = conn
            .prepare("SELECT id, name, age FROM people ORDER BY id ASC")
            .unwrap();
        let people_iter = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .unwrap();

        // Collect results
        let mut results = Vec::new();
        for person in people_iter {
            results.push(person.unwrap());
        }

        assert_eq!(results.len(), 2); // Assert on the number of rows
    }
}
