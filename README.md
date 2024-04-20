# Randas
A dataframe library in rust

This is a dataframe library in rust, i am converting one by one all the functions which i have been using in python panadas to rust.
Lack of creativity lead to the name randas, as in pandas in rust.




## Description

At its core, the library features the `DataFrame` struct, which organizes data into rows and columns. 
It supports a wide range of data types, such as :
- Integer(i64),
- Float(f64),
- Boolean(bool),
- String(String),

### Features

1. Dataframe functions: get, get_index, sum, mean, max, variance, std_dev, min, max, shape, loc, iloc etc.
2. read functions: read_csv, read_excel, read_json, read_sql
3. write functions: to_csv, to_excel, to_json, to_sql

## Installation

### Step 1: Install Rust on Windows

1. **Download Rust Installer:**
   - Visit the official Rust website at [rust-lang.org](https://www.rust-lang.org/tools/install) and download the `rustup-init.exe` installer for Windows.

2. **Run the Installer:**
   - Open the downloaded `rustup-init.exe` file and follow the on-screen instructions to install Rust. Make sure to allow `rustup` to add Rust to your system PATH.

3. **Verify the Installation:**
   - Open a new command prompt or a terminal in vscode and run `rustc --version` to check if Rust has been installed correctly.

### Step 2: Clone the Repository

1. Open the Git bash and execute: `git clone `
2. Navigate into the project directory: `cd your-repo-name`
3. **Build the Project:**
   - Compile the project with `cargo build`
4. **Run the Project (Optional):**
   - Execute the application (if applicable) with `cargo run`

## Example Usage

To use `randas` after cloning the git repo, follow this example:

```rust
// main.rs
use randas::io::{read_excel, to_excel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to your Excel file
    let input_file_path = "path_to_your_excel.xlsx";
    
    // Read the Excel file into a DataFrame
    let df = read_excel(input_file_path)?;

    // Display the DataFrame to the console
    println!("DataFrame read from Excel:");
    println!("{}", df);

    // Modify the DataFrame or use it as demonstrated
    // For example, add a new column or filter rows

    // Path for the output Excel file
    let output_file_path = "output_data.xlsx";

    // Write the DataFrame back to another Excel file
    to_excel(&df, output_file_path)?;

    Ok(())
}


to create a sample dataframe using hashmap and randas
/// ID	    Name          Score
/// 1	      Alice         3.5
/// 2	      Bob           4.0
/// 3	      Charlie       2.5
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
