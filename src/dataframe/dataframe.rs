// dataframe/dataframe.rs

use bincode;
use chrono::{DateTime, Utc};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};

/// Enum representing different data types a DataFrame can hold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DataFrameValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    DateTime(DateTime<Utc>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataFrame {
    pub columns: Vec<String>,                               // Column names
    pub data: HashMap<String, Vec<Option<DataFrameValue>>>, // Columnar data storage
    index: Vec<String>, // this will be generated automatically, cause i dont want the user to touch it at all, just read it, not WRITING it
}

impl DataFrame {
    /// Creates a new DataFrame with specified columns and columnar data,
    /// automatically generating a sequential index.
    pub fn new(
        column_data: HashMap<String, Vec<Option<DataFrameValue>>>,
        columns: Vec<String>,
    ) -> Result<Self, &'static str> {
        // Ensure all columns have the same length
        let lengths: Vec<usize> = column_data.values().map(|col| col.len()).collect();
        if lengths.windows(2).any(|w| w[0] != w[1]) {
            return Err("All columns must have the same length.");
        }

        // Generate a sequential index based on the length of the columns
        let num_rows = lengths.get(0).cloned().unwrap_or(0);
        let index = (0..num_rows)
            .map(|i| i.to_string())
            .collect::<Vec<String>>();

        Ok(DataFrame {
            data: column_data,
            columns,
            index,
        })
    }

    // user can generate a dataframe easily
    // Function to generate a new DataFrame based on column names and row values
    pub fn from_values(
        column_names: Vec<&str>,
        row_values: Vec<Vec<DataFrameValue>>,
    ) -> Result<Self, &'static str> {
        if row_values.is_empty() || row_values[0].len() != column_names.len() {
            return Err("Row values must match the number of columns");
        }

        let mut data: HashMap<String, Vec<Option<DataFrameValue>>> = HashMap::new();
        let num_rows = row_values.len();

        // Initialize columns in the HashMap
        for &col_name in &column_names {
            data.insert(col_name.to_string(), vec![None; num_rows]);
        }

        // Populate the data
        for (row_idx, row) in row_values.into_iter().enumerate() {
            for (col_idx, value) in row.into_iter().enumerate() {
                let col_name = &column_names[col_idx];
                if let Some(column) = data.get_mut(*col_name) {
                    column[row_idx] = Some(value);
                }
            }
        }

        // Generate index
        let index = (0..num_rows).map(|i| i.to_string()).collect();

        Ok(DataFrame {
            columns: column_names
                .into_iter()
                .map(|name| name.to_string())
                .collect(),
            data,
            index,
        })
    }

    /// Returns a column hashmap using the column name as argument
    /// # Arguments
    ///
    /// * `column_name` - The name of the column to retrieve.
    ///
    /// # Returns
    ///
    /// An `Option<&Vec<Option<DataFrameValue>>>` representing the column's data.
    /// Returns `None` if the column does not exist.
    pub fn column(&self, column_name: &str) -> Option<&Vec<Option<DataFrameValue>>> {
        self.data.get(column_name)
    }

    /// Returns a single value from the DataFrame by specifying a row and a column name.
    pub fn get(&self, row: usize, column_name: &str) -> Option<&Option<DataFrameValue>> {
        // access the column by name
        if let Some(column) = self.data.get(column_name) {
            // If the column exists, return the value at the specified row index if it exists,
            // otherwise return a reference to None indicating the row doesn't exist within the column
            Some(column.get(row).unwrap_or(&None))
        } else {
            // If the column does not exist, return None
            None
        }
    }

    /// to get dataframe index
    pub fn get_index(&self) -> &[String] {
        &self.index
    }

    /// to get a value
    pub fn get_value(&self, row_index: usize, column_name: &str) -> Option<&DataFrameValue> {
        self.data
            .get(column_name)
            .and_then(|col| col.get(row_index))
            .and_then(|val| val.as_ref())
    }

    /// Adds a new column to the DataFrame. The index remains unchanged as it's row-specific.
    pub fn add_column(
        &mut self,
        column_name: String,
        column_data: Vec<Option<DataFrameValue>>,
    ) -> Result<(), &'static str> {
        // First, check if the new column's length matches the expected number of rows.
        let expected_rows = self.data.values().next().map_or(0, Vec::len);

        if expected_rows != 0 && expected_rows != column_data.len() {
            return Err("Column length does not match the number of rows in the DataFrame.");
        }

        // Insert the new column into the data HashMap.
        let inserted = self.data.insert(column_name.clone(), column_data).is_none();

        // Only update the columns vector if the column was newly inserted,
        // not when an existing column was replaced.
        if inserted {
            self.columns.push(column_name);
        }

        // If this is the first column being added, initialize the index based on its length.
        if self.index.is_empty() {
            let column_length = self.data.values().next().unwrap().len();
            self.index = (0..column_length).map(|i| i.to_string()).collect();
        }

        Ok(())
    }

    /// Removes a column from the DataFrame by its name.
    pub fn remove_column(&mut self, column_name: &str) -> Result<(), &'static str> {
        // Check if the column exists by removing it from the HashMap.
        // `HashMap::remove` returns `None` if the key was not found.
        if self.data.remove(column_name).is_none() {
            return Err("Column name not found.");
        }

        // Remove the column name from the list of column names.
        // This step is necessary to keep the `columns` vector in sync with the actual data.
        if let Some(pos) = self.columns.iter().position(|col| col == column_name) {
            self.columns.remove(pos);
        }

        Ok(())
    }

    /// Calculate the sum of a specific column, handling different numeric types.
    /// If the column contains non-numeric data, this will return None.
    pub fn sum(&self, column_name: &str) -> Option<f64> {
        self.data.get(column_name).map_or(None, |column| {
            // Initialize accumulator as 0.0
            // Use `fold` to iterate over each value in the column
            let sum = column.iter().fold(0.0, |acc, value| match value {
                // Add the value to the accumulator if it's a Float
                Some(DataFrameValue::Float(f)) => acc + f,
                // Convert the Integer to f64 and add it to the accumulator
                Some(DataFrameValue::Integer(i)) => acc + *i as f64,
                // If the value is not numeric or is None, just return the current accumulator
                _ => acc,
            });
            // for an empty column; 0.0 makes more sense as the sum of an empty set.
            Some(sum)
        })
    }

    /// Calculates the mean (average) of the numeric values in a specified column.
    ///
    /// # Arguments
    ///
    /// * `column_name` - The name of the column for which to calculate the mean.
    ///
    /// # Returns
    ///
    /// An `Option<f64>` representing the mean of the column. Returns `None` if the column
    /// does not exist, cannot be converted to numeric values, or if the column contains no data.
    pub fn mean(&self, column_name: &str) -> Option<f64> {
        self.data.get(column_name).map_or(None, |column| {
            let (sum, count) = column.iter().fold((0.0, 0), |(acc_sum, acc_count), value| {
                match value {
                    Some(DataFrameValue::Float(f)) => (acc_sum + f, acc_count + 1),
                    Some(DataFrameValue::Integer(i)) => (acc_sum + *i as f64, acc_count + 1),
                    _ => (acc_sum, acc_count), // Ignore non-numeric values
                }
            });

            if count > 0 {
                Some(sum / count as f64)
            } else {
                None
            }
        })
    }

    /// Finds the maximum numeric value in each column of the DataFrame.
    pub fn max(&self) -> Vec<Option<f64>> {
        self.columns
            .iter()
            .map(|column_name| {
                self.data.get(column_name).and_then(|column_data| {
                    // Convert both Integer and Float values to f64 for comparison,
                    // ignoring non-numeric values and empty columns.
                    let numeric_values: Vec<f64> = column_data
                        .iter()
                        .filter_map(|value| match value {
                            Some(DataFrameValue::Float(f)) => Some(*f),
                            Some(DataFrameValue::Integer(i)) => Some(*i as f64),
                            _ => None, // Non-numeric values are ignored
                        })
                        .collect();

                    if numeric_values.is_empty() {
                        // If there are no numeric values in the column, return None
                        None
                    } else {
                        // Use max_by with partial_cmp to find the maximum numeric value,
                        // handling potential NaN values safely
                        numeric_values
                            .iter()
                            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .copied()
                    }
                })
            })
            .collect()
    }

    /// Calculates the variance of the numeric values in a specified column.
    /// Variance is the average of the squared differences from the mean.
    ///
    /// # Arguments
    ///
    /// * `column_name` - The name of the column for which to calculate the variance.
    ///
    /// # Returns
    ///
    /// An `Option<f64>` representing the variance of the column. Returns `None` if the column
    /// does not exist, cannot be converted to numeric values, or if the column contains less than
    /// two data points.
    pub fn variance(&self, column_name: &str) -> Option<f64> {
        let column_data = self.data.get(column_name)?;
        let mean = self.mean(column_name)?;
        let sum_of_squared_diffs: f64 = column_data
            .iter()
            .filter_map(|v| match v {
                Some(DataFrameValue::Float(value)) => Some((*value - mean).powi(2)),
                Some(DataFrameValue::Integer(value)) => Some((*value as f64 - mean).powi(2)),
                _ => None,
            })
            .sum();
        let count = column_data
            .iter()
            .filter(|v| {
                matches!(
                    v,
                    Some(DataFrameValue::Float(_)) | Some(DataFrameValue::Integer(_))
                )
            })
            .count();

        if count > 1 {
            Some(sum_of_squared_diffs / (count as f64 - 1.0))
        } else {
            None
        }
    }

    /// Calculates the standard deviation of the numeric values in a specified column.
    /// The standard deviation is a measure of the amount of variation or dispersion of a set of values.
    ///
    /// # Arguments
    ///
    /// * `column_name` - The name of the column for which to calculate the standard deviation.
    ///
    /// # Returns
    ///
    /// An `Option<f64>` representing the standard deviation of the column. Returns `None` if the column
    /// does not exist, cannot be converted to numeric values, or if the column contains less than
    /// two data points.
    pub fn std_dev(&self, column_name: &str) -> Option<f64> {
        self.variance(column_name).map(|variance| variance.sqrt())
    }

    /// Finds the minimum value of each numeric column in the DataFrame.
    pub fn min(&self) -> Vec<Option<f64>> {
        self.columns
            .iter()
            .map(|column_name| {
                // Directly access the column's data using its name
                if let Some(column_data) = self.data.get(column_name) {
                    column_data
                        .iter()
                        .filter_map(|value| {
                            match value {
                                Some(DataFrameValue::Float(f)) => Some(*f),
                                Some(DataFrameValue::Integer(i)) => Some(*i as f64),
                                _ => None,
                            }
                            // Use min_by with partial_cmp to find the minimum value,
                            // handling potential NaN values safely
                        })
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                } else {
                    None // If for some reason the column data is not found (which shouldn't happen), return None
                }
            })
            .collect()
    }

    /// Returns the shape of a dataframe as (rows, columns)
    pub fn shape(&self) -> (usize, usize) {
        let rows = self.data.values().next().map_or(0, |v| v.len());
        let columns = self.columns.len();
        (rows, columns)
    }

    /// Counts the non-null (non-`None`) values of each column in the DataFrame.
    pub fn count(&self) -> Vec<usize> {
        self.columns
            .iter()
            .map(|column_name| {
                // Access the column's data directly using the column name
                if let Some(column_data) = self.data.get(column_name) {
                    // Count non-`None` values in the column
                    column_data.iter().filter(|value| value.is_some()).count()
                } else {
                    // If for some reason the column data is not found, return a count of 0
                    0
                }
            })
            .collect()
    }

    /// Collects the values of a specific column from the DataFrame.
    ///
    /// # Arguments
    ///
    /// * `column_name` - The name of the column to collect values from.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec` of column values if successful, or an error message if the column does not exist.
    pub fn collect_column_values(
        &self,
        column_name: &str,
    ) -> Result<Vec<Option<DataFrameValue>>, &'static str> {
        // Directly access the column's data using the column name from the HashMap
        match self.data.get(column_name) {
            Some(column_data) => Ok(column_data.clone()), // Clone the column's data Vec to return
            None => Err("Column name not found."), // If the column name doesn't exist in the data HashMap
        }
    }

    /// Returns a subset of the DataFrame based on row indices and an optional list of column names.
    /// If no column names are provided, all columns are included.
    pub fn loc(
        &self,
        row_indices: &[usize],
        column_names: Option<&[&str]>,
    ) -> Result<DataFrame, &'static str> {
        let mut new_data = HashMap::new();
        let mut new_columns = Vec::new();

        let cols_to_use = match column_names {
            Some(names) => names
                .iter()
                .map(|&n| n.to_string())
                .collect::<Vec<String>>(),
            None => self.columns.clone(),
        };

        // Validate and collect requested columns or all columns if None
        for name in cols_to_use.iter() {
            if let Some(column_data) = self.data.get(name) {
                let filtered_data: Vec<Option<DataFrameValue>> = row_indices
                    .iter()
                    .filter_map(|&idx| column_data.get(idx).cloned())
                    .collect();

                if filtered_data.len() != row_indices.len() {
                    return Err("One or more row indices out of bounds.");
                }

                new_data.insert(name.to_string(), filtered_data);
                new_columns.push(name.to_string());
            } else {
                return Err("One or more column names not found.");
            }
        }

        // Generate new index based on row_indices
        let new_index = row_indices
            .iter()
            .filter_map(|&i| self.index.get(i).cloned())
            .collect();

        Ok(DataFrame {
            data: new_data,
            columns: new_columns,
            index: new_index,
        })
    }

    // `iloc` method to access data by integer location.
    pub fn iloc(
        &self,
        row_idx: usize,
        col_idx: usize,
    ) -> Result<&Option<DataFrameValue>, &'static str> {
        // Check if the row index is within the bounds of the DataFrame's rows.
        if row_idx >= self.index.len() {
            return Err("Row index is out of bounds.");
        }

        // Retrieve the column name using the column index.
        let column_name = self
            .columns
            .get(col_idx)
            .ok_or("Column index is out of bounds.")?;

        // Access the data using the column name and row index.
        let column_data = self.data.get(column_name).ok_or("Column not found.")?;

        // Retrieve the value at the specified row index within the column.
        let value = column_data
            .get(row_idx)
            .ok_or("Row index is out of bounds within the column.")?;

        Ok(value)
    }

    // Function to filter rows based on boolean condition
    pub fn boolean_index(&self, condition: Vec<bool>) -> Result<DataFrame, &'static str> {
        if condition.len() != self.data.len() {
            return Err("Condition length does not match number of rows in DataFrame.");
        }

        // store the filtered data here
        let mut filtered_data: HashMap<String, Vec<Option<DataFrameValue>>> = HashMap::new();

        // Initialize the filtered data with empty vectors for each column
        for column in &self.columns {
            filtered_data.insert(column.clone(), Vec::new());
        }

        // Iterate over each row and apply the condition
        for (i, &include) in condition.iter().enumerate() {
            if include {
                // If the condition is true, include this row in the filtered DataFrame
                for column in &self.columns {
                    let column_values = filtered_data.get_mut(column).unwrap();
                    let value = self.data.get(column).unwrap()[i].clone();
                    column_values.push(value);
                }
            }
        }

        // Construct the new DataFrame
        Ok(DataFrame {
            columns: self.columns.clone(),
            data: filtered_data,
            index: self
                .index
                .iter()
                .enumerate()
                .filter_map(|(i, idx)| {
                    if condition[i] {
                        Some(idx.clone())
                    } else {
                        None
                    }
                })
                .collect(),
        })
    }

    /// Filters the DataFrame based on a vector of boolean values indicating which rows to keep.
    pub fn filter_by_condition(&self, condition_results: Vec<bool>) -> DataFrame {
        let mut filtered_data = HashMap::new();

        for (name, column) in &self.data {
            let filtered_column: Vec<Option<DataFrameValue>> = column
                .iter()
                .zip(condition_results.iter())
                .filter_map(|(value, &keep)| if keep { Some(value.clone()) } else { None })
                .collect();
            filtered_data.insert(name.clone(), filtered_column);
        }

        let filtered_columns = self.columns.clone();
        DataFrame::new(filtered_data, filtered_columns).unwrap()
    }

    /// Applies a condition to a column and returns a vector indicating which rows meet the condition.
    /// This is more of a helper function for filtering
    pub fn apply_condition<F>(
        &self,
        column_name: &str,
        condition: F,
    ) -> Result<Vec<bool>, &'static str>
    where
        F: Fn(&DataFrameValue) -> bool,
    {
        let column_data = self.data.get(column_name).ok_or("Column not found")?;
        let condition_results = column_data
            .iter()
            .map(|value| match value {
                Some(v) => condition(v),
                None => false,
            })
            .collect();

        Ok(condition_results)
    }

    /// Returns the first `n` rows of the DataFrame.
    pub fn head(&self, n: Option<usize>) -> DataFrame {
        let n = n.unwrap_or(5);

        // Initialize a new HashMap for the new data
        let mut new_data = HashMap::new();

        // For each column, take the first `n` values
        for (name, values) in &self.data {
            let column_head = values.iter().take(n).cloned().collect::<Vec<_>>();
            new_data.insert(name.clone(), column_head);
        }

        DataFrame {
            columns: self.columns.clone(),
            data: new_data,
            index: self.index.iter().take(n).cloned().collect(),
        }
    }

    /// Returns the last `n` rows of the DataFrame.
    pub fn tail(&self, n: Option<usize>) -> DataFrame {
        let n = n.unwrap_or(5);

        // Initialize a new HashMap for the new data
        let mut new_data = HashMap::new();

        // For each column, take the last `n` values
        for (name, values) in &self.data {
            let len = values.len();
            let start = if n >= len { 0 } else { len - n };
            let column_tail = values[start..].to_vec();
            new_data.insert(name.clone(), column_tail);
        }

        DataFrame {
            columns: self.columns.clone(),
            data: new_data,
            index: if n >= self.index.len() {
                self.index.clone()
            } else {
                self.index[self.index.len() - n..].to_vec()
            },
        }
    }

    /// Generates a descriptive statistics DataFrame for the numerical columns.
    pub fn describe(&self) -> Result<DataFrame, &'static str> {
        let mut descriptions: HashMap<String, Vec<Option<DataFrameValue>>> = HashMap::new();

        // Prepare description columns
        let description_columns = vec![
            "Count", "Mean", "Std Dev", "Min", "25%", "50%", "75%", "Max",
        ];

        for (name, column) in &self.data {
            let numerical_data: Vec<f64> = column
                .iter()
                .filter_map(|value| match value {
                    Some(DataFrameValue::Integer(i)) => Some(*i as f64),
                    Some(DataFrameValue::Float(f)) => Some(*f),
                    _ => None,
                })
                .collect();

            if !numerical_data.is_empty() {
                let count = numerical_data.len() as f64;
                let sum: f64 = numerical_data.iter().sum();
                let mean = sum / count;
                let std_dev = (numerical_data
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / count)
                    .sqrt();

                // Quartiles
                let mut sorted = numerical_data.clone();
                sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let median = sorted[sorted.len() / 2];
                let q1 = sorted[sorted.len() / 4];
                let q3 = sorted[3 * sorted.len() / 4];
                let min = *sorted.first().unwrap();
                let max = *sorted.last().unwrap();

                descriptions.insert(
                    name.clone(),
                    vec![
                        Some(DataFrameValue::Float(count)),
                        Some(DataFrameValue::Float(mean)),
                        Some(DataFrameValue::Float(std_dev)),
                        Some(DataFrameValue::Float(min)),
                        Some(DataFrameValue::Float(q1)),
                        Some(DataFrameValue::Float(median)),
                        Some(DataFrameValue::Float(q3)),
                        Some(DataFrameValue::Float(max)),
                    ],
                );
            }
        }

        // Create the descriptive DataFrame
        DataFrame::new(
            descriptions,
            description_columns.iter().map(|s| s.to_string()).collect(),
        )
    }

    fn min_max_f64(values: &[f64]) -> (f64, f64) {
        values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                let min = if val.is_nan() { min } else { val.min(min) };
                let max = if val.is_nan() { max } else { val.max(max) };
                (min, max)
            })
    }

    /// Concatenates two DataFrames along the specified axis.
    pub fn concat(&self, other: &DataFrame, axis: usize) -> Result<DataFrame, &'static str> {
        match axis {
            0 => self.concat_rows(other),
            1 => self.concat_columns(other),
            _ => Err("Invalid axis. Use 0 for rows or 1 for columns."),
        }
    }

    /// Concatenates two DataFrames along rows (axis=0).
    fn concat_rows(&self, other: &DataFrame) -> Result<DataFrame, &'static str> {
        if self.columns != other.columns {
            return Err("DataFrames have different columns. Cannot concatenate by rows.");
        }

        let mut new_data = HashMap::new();
        for column in &self.columns {
            let mut column_data = self
                .data
                .get(column)
                .ok_or("Column not found in self DataFrame")?
                .clone();
            column_data.extend(
                other
                    .data
                    .get(column)
                    .ok_or("Column not found in other DataFrame")?
                    .clone(),
            );
            new_data.insert(column.clone(), column_data);
        }

        let mut new_index = self.index.clone();
        let offset = self.index.len();
        new_index.extend(
            other
                .index
                .iter()
                .enumerate()
                .map(|(i, val)| format!("{}-{}", offset + i, val)),
        );

        Ok(DataFrame {
            columns: self.columns.clone(),
            data: new_data,
            index: new_index,
        })
    }

    /// Concatenates two DataFrames along columns (axis=1).
    fn concat_columns(&self, other: &DataFrame) -> Result<DataFrame, &'static str> {
        if self.data.len() != other.data.len() {
            return Err(
                "DataFrames have different numbers of rows. Cannot concatenate by columns.",
            );
        }

        let mut new_data = self.data.clone();
        for (column, data) in &other.data {
            if new_data.contains_key(column) {
                return Err("Cannot concatenate DataFrames with overlapping column names.");
            }
            new_data.insert(column.clone(), data.clone());
        }

        let mut new_columns = self.columns.clone();
        new_columns.extend(other.columns.iter().cloned());
        let new_index = self.index.clone(); //  row counts are identical.

        Ok(DataFrame {
            columns: new_columns,
            data: new_data,
            index: new_index,
        })
    }

    /// Extract numeric values from a specified column as a `Vec<f64>`.
    /// Non-numeric values are ignored.
    pub fn extract_numeric_values(&self, column_name: &str) -> Result<Vec<f64>, Box<dyn Error>> {
        let column_data = self
            .data
            .get(column_name)
            .ok_or_else(|| format!("Column name {} not found", column_name))?;

        let numeric_values: Vec<f64> = column_data
            .iter()
            .filter_map(|value| match value {
                Some(DataFrameValue::Float(val)) => Some(*val),
                Some(DataFrameValue::Integer(val)) => Some(*val as f64),
                _ => None, // Ignore non-numeric values
            })
            .collect();

        Ok(numeric_values)
    }

    /// to serialize (or "pickle" in python term) a Dataframe into a binary format
    pub fn save_to_file(&self, file_path: &str) -> io::Result<()> {
        let encoded: Vec<u8> = bincode::serialize(self).unwrap();
        let mut file = File::create(file_path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    /// To deserialize (or "unpickle" in python term) a DataFrame from a binary format stored on disk
    pub fn read_from_file(file_path: &str) -> io::Result<Self> {
        let mut file = File::open(file_path)?;
        let mut encoded = Vec::new();
        file.read_to_end(&mut encoded)?;
        let dataframe: DataFrame = bincode::deserialize(&encoded[..]).unwrap();
        Ok(dataframe)
    }

    /// Removes rows with any `None` values.
    pub fn dropna(&mut self) -> &mut Self {
        let num_rows = self.index.len();
        let mut drop_indices = Vec::new();

        // Identify rows to drop
        for i in 0..num_rows {
            for column in self.data.values() {
                if column[i].is_none() {
                    drop_indices.push(i);
                    break; // Move to the next row if a None value is found
                }
            }
        }

        // Reverse the indices to avoid shifting positions affecting removal
        drop_indices.reverse();
        for &index in &drop_indices {
            for column in self.data.values_mut() {
                column.remove(index);
            }
            self.index.remove(index);
        }

        self
    }

    // Fills `None` values with a specified replacement value.
    pub fn fillna(&mut self, column_name: &str, fill_value: DataFrameValue) {
        if let Some(column) = self.data.get_mut(column_name) {
            for value in column.iter_mut() {
                if value.is_none() {
                    *value = Some(fill_value.clone()); // Clone the fill_value for each None found
                }
            }
        }
    }

    // A simple query function for equality
    pub fn query(&self, query: &str) -> Self {
        let parts: Vec<&str> = query.split("==").map(|part| part.trim()).collect();
        if parts.len() != 2 {
            panic!("Invalid query format. Use 'column_name == value'.");
        }
        let (column, value) = (parts[0], parts[1]);

        // Check if the column exists
        if !self.columns.contains(&column.to_string()) {
            panic!("Column '{}' does not exist.", column);
        }

        // Determine the value type
        let value = if let Ok(int_val) = value.parse::<i64>() {
            DataFrameValue::Integer(int_val)
        } else if let Ok(float_val) = value.parse::<f64>() {
            DataFrameValue::Float(float_val)
        } else {
            // Assume string for simplicity; real implementation should handle more cases
            DataFrameValue::String(value.to_string())
        };

        let filtered_indices: Vec<String> = self.data[column]
            .iter()
            .enumerate()
            .filter_map(|(index, data_val)| match (data_val, &value) {
                (Some(DataFrameValue::Integer(int_val)), DataFrameValue::Integer(query_val))
                    if int_val == query_val =>
                {
                    Some(self.index[index].clone())
                }
                (Some(DataFrameValue::Float(float_val)), DataFrameValue::Float(query_val))
                    if float_val == query_val =>
                {
                    Some(self.index[index].clone())
                }
                (Some(DataFrameValue::String(str_val)), DataFrameValue::String(query_val))
                    if str_val == query_val =>
                {
                    Some(self.index[index].clone())
                }
                _ => None,
            })
            .collect();

        // Construct a new DataFrame with filtered rows
        let mut new_data: HashMap<String, Vec<Option<DataFrameValue>>> = HashMap::new();
        for column in &self.columns {
            let column_data: Vec<Option<DataFrameValue>> = filtered_indices
                .iter()
                .map(|index| {
                    let row = self.index.iter().position(|r| r == index).unwrap();
                    self.data[column][row].clone()
                })
                .collect();
            new_data.insert(column.clone(), column_data);
        }

        DataFrame {
            columns: self.columns.clone(),
            data: new_data,
            index: filtered_indices,
        }
    }

    /// scatter plot function implementation using dataframe
    pub fn scatter_plot(
        &self,
        x_col_name: &str,
        y_col_name: &str,
        output_file: &str,
        plot_title: Option<&str>,
        _x_axis_label: Option<&str>,
        _y_axis_label: Option<&str>,
    ) -> Result<(), Box<dyn Error>> {
        // Previous code unchanged...

        // Extract and collect values, ensuring they're owned (copied) values
        let x_values: Vec<f64> = self.extract_numeric_values(x_col_name)?;
        let y_values: Vec<f64> = self.extract_numeric_values(y_col_name)?;

        // Ensure both columns have the same number of data points
        assert_eq!(
            x_values.len(),
            y_values.len(),
            "Columns have different lengths"
        );

        // Plotting logic
        let root_area = BitMapBackend::new(output_file, (640, 480)).into_drawing_area();
        root_area.fill(&WHITE)?;

        let x_min_max = Self::min_max_f64(&x_values);
        let y_min_max = Self::min_max_f64(&y_values);

        let mut chart = ChartBuilder::on(&root_area)
            .caption(plot_title.unwrap_or("Scatter Plot"), ("sans-serif", 50.))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min_max.0..x_min_max.1, y_min_max.0..y_min_max.1)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(
            x_values
                .into_iter()
                .zip(y_values.into_iter())
                .map(|(x, y)| {
                    Circle::new((x, y), 5, RED.filled()) // Use RED.filled() directly
                }),
        )?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root_area.present()?;

        Ok(())
    }
}

impl PartialEq for DataFrame {
    fn eq(&self, other: &Self) -> bool {
        // First, check if the columns match
        if self.columns != other.columns {
            return false;
        }

        // Next, check if the data matches
        if self.data.len() != other.data.len() {
            return false;
        }

        for (row_a, row_b) in self.data.iter().zip(other.data.iter()) {
            if row_a != row_b {
                return false;
            }
        }

        true
    }
}

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let column_width = 20; // Set a constant for uniform column width

        // Display the column names with a header
        write!(f, "{:<width$}", "Index", width = column_width)?;
        for column in &self.columns {
            write!(f, "{:<width$}", column, width = column_width)?;
        }
        writeln!(f)?;
        writeln!(f, "{}", "-".repeat((self.columns.len() + 1) * column_width))?; // Separator line including Index

        //  the length of rows can be determined from the first column's data
        let row_count = self.data.get(&self.columns[0]).map_or(0, |col| col.len());

        for row_index in 0..row_count {
            write!(f, "{:<width$}", row_index, width = column_width)?;
            for column in &self.columns {
                let value = self
                    .data
                    .get(column)
                    .and_then(|col| col.get(row_index)) // This gets &Option<DataFrameValue>
                    .unwrap_or(&None); // Maintain reference here to match types

                match *value {
                    Some(DataFrameValue::Integer(val)) => {
                        write!(f, "{:<width$}", val, width = column_width)?
                    }
                    Some(DataFrameValue::Float(val)) => {
                        write!(f, "{:<width$.2}", val, width = column_width)?
                    }
                    Some(DataFrameValue::Boolean(val)) => {
                        write!(f, "{:<width$}", val, width = column_width)?
                    }
                    Some(DataFrameValue::String(ref val)) => {
                        write!(f, "{:<width$}", val, width = column_width)?
                    }
                    Some(DataFrameValue::DateTime(ref val)) => write!(
                        f,
                        "{:<width$}",
                        val.format("%Y-%m-%d %H:%M:%S"),
                        width = column_width
                    )?,
                    None => write!(f, "{:<width$}", "NA", width = column_width)?,
                };
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::dataframe::dataframe::{DataFrame, DataFrameValue};
    use std::collections::HashMap;

    /// this is the sample dataframe, that i can use throughout the unit testcases
    ///     	ID	    Name	    Score
    /// 	    1	    Alice	    3.5
    /// 	    2	    Bob	        4.0
    /// 	    3	    Charlie	    2.5
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
    fn test_dataframe_from_values() {
        // Define column names and rows values for the test DataFrame
        let column_names = vec!["ID", "Name", "Age"];
        let row_values = vec![
            vec![
                DataFrameValue::Integer(1),
                DataFrameValue::String("Alice".to_string()),
                DataFrameValue::Integer(30),
            ],
            vec![
                DataFrameValue::Integer(2),
                DataFrameValue::String("Bob".to_string()),
                DataFrameValue::Integer(25),
            ],
            vec![
                DataFrameValue::Integer(3),
                DataFrameValue::String("Charlie".to_string()),
                DataFrameValue::Integer(35),
            ],
        ];

        // Create the DataFrame
        let df =
            DataFrame::from_values(column_names, row_values).expect("Failed to create DataFrame");

        // Check if the DataFrame contains the expected number of rows and columns
        assert_eq!(df.columns.len(), 3, "DataFrame should have 3 columns");
        assert_eq!(
            df.data.get("ID").unwrap().len(),
            3,
            "ID column should have 3 values"
        );
        assert_eq!(
            df.data.get("Name").unwrap().len(),
            3,
            "Name column should have 3 values"
        );
        assert_eq!(
            df.data.get("Age").unwrap().len(),
            3,
            "Age column should have 3 values"
        );

        // Verify the content of the DataFrame
        assert_eq!(
            *df.data.get("ID").unwrap().get(0).unwrap(),
            Some(DataFrameValue::Integer(1))
        );
        assert_eq!(
            *df.data.get("Name").unwrap().get(1).unwrap(),
            Some(DataFrameValue::String("Bob".to_string()))
        );
        assert_eq!(
            *df.data.get("Age").unwrap().get(2).unwrap(),
            Some(DataFrameValue::Integer(35))
        );

        // Ensure the indices are correct
        assert_eq!(df.index, vec!["0", "1", "2"]);
    }

    #[test]
    fn test_get_method() {
        // Setup a sample DataFrame
        let df = setup_test_dataframe();

        // Get a known value
        if let Some(value) = df.get(1, "Name") {
            assert_eq!(*value, Some(DataFrameValue::String("Bob".to_string())));
        } else {
            panic!("Expected value not found");
        }

        // Get a value using a non-existent column, , so expectation is that it should fail
        assert!(
            df.get(1, "NonExistentColumn").is_none(),
            "Expected None for a non-existent column"
        );

        // Get a value using an out-of-bounds row index, so expectation is that it should fail
        assert!(
            matches!(df.get(99, "Name"), Some(None)),
            "Expected Some(None) for an out-of-bounds row index"
        );
    }

    #[test]
    fn test_column_function() {
        let df = setup_test_dataframe();

        // Test retrieving each column
        let id_column = df.column("ID").unwrap();
        assert_eq!(
            id_column,
            &vec![
                Some(DataFrameValue::Integer(1)),
                Some(DataFrameValue::Integer(2)),
                Some(DataFrameValue::Integer(3))
            ]
        );

        let name_column = df.column("Name").unwrap();
        assert_eq!(
            name_column,
            &vec![
                Some(DataFrameValue::String("Alice".to_string())),
                Some(DataFrameValue::String("Bob".to_string())),
                Some(DataFrameValue::String("Charlie".to_string()))
            ]
        );

        let score_column = df.column("Score").unwrap();
        assert_eq!(
            score_column,
            &vec![
                Some(DataFrameValue::Float(3.5)),
                Some(DataFrameValue::Float(4.0)),
                Some(DataFrameValue::Float(2.5))
            ]
        );

        // Test for a non-existent column
        assert!(df.column("Nonexistent").is_none());
    }

    #[test]
    fn test_get_index() {
        let df = setup_test_dataframe();

        let index = df.get_index();
        let expected_index = &["0", "1", "2"];
        assert_eq!(
            index, expected_index,
            "The index should correctly match the expected values"
        );
    }

    #[test]
    fn test_get_value() {
        let df = setup_test_dataframe();

        // Test valid values
        assert_eq!(
            df.get_value(0, "Name"),
            Some(&DataFrameValue::String("Alice".to_string())),
            "Should retrieve 'Alice' for first Name"
        );
        assert_eq!(
            df.get_value(1, "Score"),
            Some(&DataFrameValue::Float(4.0)),
            "Should retrieve 4.0 for second Score"
        );

        // Test invalid column name
        assert_eq!(
            df.get_value(1, "Unknown"),
            None,
            "Should return None for an unknown column"
        );

        // Test with index out of bounds
        assert_eq!(
            df.get_value(5, "Name"),
            None,
            "Should return None for an out-of-bounds index"
        );
    }

    #[test]
    fn test_add_column() {
        let mut df = setup_test_dataframe(); // DataFrame

        // Add a new column that correctly matches the number of rows.
        let result = df.add_column(
            "NewColumn".to_string(),
            vec![
                Some(DataFrameValue::Float(1.0)),
                Some(DataFrameValue::Float(2.0)),
                Some(DataFrameValue::Float(11.0)),
            ],
        );
        assert!(result.is_ok());

        // check if the new column was added.
        assert_eq!(df.columns.len(), 4); //  there were initially 3 columns.

        // Verify the new column's data matches the expected length.
        let new_column_data = df.data.get("NewColumn").unwrap();
        assert_eq!(new_column_data.len(), 3); // Verifying the new column has 3 entries.

        // add a column with mismatching length should fail.
        let mismatch_result = df.add_column(
            "Mismatched".to_string(),
            vec![Some(DataFrameValue::Float(3.0))], // Incorrect length
        );
        assert!(mismatch_result.is_err());
    }

    #[test]
    fn test_remove_column() {
        let mut df = setup_test_dataframe(); // Setup DataFrame

        // DataFrame initially has 3 columns.
        assert_eq!(df.columns.len(), 3);
        assert!(df.data.contains_key("Score"));
        assert!(df.remove_column("Score").is_ok());

        // check if the column has been removed.
        assert_eq!(df.columns.len(), 2); // Now there should be 2 columns.
        assert!(!df.columns.contains(&"Score".to_string()));
        assert!(!df.data.contains_key("Score")); // Ensure "Score" column no longer exists

        // check if the remaining columns still have the correct number of data entries.
        for column_data in df.data.values() {
            assert_eq!(column_data.len(), 3);
        }

        // remove a non-existent column should fail.
        assert!(df.remove_column("NonExistentColumn").is_err());
    }

    #[test]
    fn test_column_sum() {
        let df = setup_test_dataframe();

        // Test summing a numeric column
        let score_sum = df.sum("Score");
        assert_eq!(score_sum, Some(10.0)); //  the expected sum of the "Score" column is 10.0

        // Test summing a non-numeric column, which should return None
        let name_sum = df.sum("Name");
        assert_eq!(name_sum, Some(0.0));

        // Test summing a non-existent column, which should also return None
        let nonexistent_column_sum = df.sum("NonexistentColumn");
        assert_eq!(nonexistent_column_sum, None);
    }

    #[test]
    fn test_mean() {
        let df = setup_test_dataframe();

        // Test the mean of a numeric column
        assert_eq!(df.mean("Score"), Some(3.3333333333333335));

        // Test the mean of a non-numeric column, which should return None
        assert_eq!(df.mean("Name"), None);

        // Test the mean of a non-existent column, which should also return None
        assert_eq!(df.mean("NonexistentColumn"), None);
    }

    #[test]
    fn test_max() {
        let df = setup_test_dataframe();
        let max_values = df.max();

        assert_eq!(
            max_values[2],
            Some(4.0),
            "The maximum value for 'Score' column should be 4.0"
        );
    }

    #[test]
    fn test_min_function() {
        let df = setup_test_dataframe();

        // Execute the min function
        let min_values = df.min();

        // Define expected results
        assert_eq!(min_values[0], Some(1.0), "Minimum of ID should be 1.0");
        assert_eq!(
            min_values[1], None,
            "Minimum of Name should be None because it's non-numeric"
        );
        assert_eq!(min_values[2], Some(2.5), "Minimum of Score should be 2.5");
    }

    #[test]
    fn test_variance() {
        let df = setup_test_dataframe();

        let variance_score = df.variance("Score").unwrap();
        let expected_variance = 0.58333333; // This is the computed expected variance
        let tolerance = 0.001;

        assert!(
            (variance_score - expected_variance).abs() < tolerance,
            "Variance should be around 0.58333333"
        );
    }

    #[test]
    fn test_std_dev() {
        let df = setup_test_dataframe();

        // Check standard deviation calculation on the "Score" column
        let std_dev_score = df.std_dev("Score").unwrap();
        let expected_std_dev = (0.58333333f64).sqrt();
        let tolerance = 0.001;

        assert!(
            (std_dev_score - expected_std_dev).abs() < tolerance,
            "Standard deviation should be close to the expected value"
        );
    }

    #[test]
    fn test_shape() {
        let df = setup_test_dataframe();

        // Check the shape of the DataFrame
        let (rows, columns) = df.shape();
        assert_eq!(rows, 3, "There should be 3 rows in the DataFrame");
        assert_eq!(columns, 3, "There should be 3 columns in the DataFrame");

        // Test an empty DataFrame
        let empty_df =
            DataFrame::new(HashMap::new(), Vec::new()).expect("Failed to create DataFrame");
        let (empty_rows, empty_columns) = empty_df.shape();
        assert_eq!(
            empty_rows, 0,
            "There should be 0 rows in the empty DataFrame"
        );
        assert_eq!(
            empty_columns, 0,
            "There should be 0 columns in the empty DataFrame"
        );
    }

    #[test]
    fn test_count_non_null_values() {
        let df = setup_test_dataframe();

        // Execute the count function
        let counts = df.count();

        // Assuming the DataFrame is set up as described earlier
        assert_eq!(
            counts[0], 3,
            "All entries in the 'ID' column should be non-None"
        );
        assert_eq!(
            counts[1], 3,
            "All entries in the 'Name' column should be non-None"
        );
        assert_eq!(
            counts[2], 3,
            "All entries in the 'Score' column should be non-None"
        );
    }

    #[test]
    fn test_dataframe_index_generation() {
        let data: HashMap<String, Vec<Option<DataFrameValue>>> = [
            (
                "Column1".to_string(),
                vec![
                    Some(DataFrameValue::Integer(1)),
                    Some(DataFrameValue::Integer(2)),
                ],
            ),
            (
                "Column2".to_string(),
                vec![
                    Some(DataFrameValue::String("A".to_string())),
                    Some(DataFrameValue::String("B".to_string())),
                ],
            ),
        ]
        .iter()
        .cloned()
        .collect();

        let df = DataFrame::new(data, vec!["Column1".to_string(), "Column2".to_string()]).unwrap();

        assert_eq!(df.index, vec!["0".to_string(), "1".to_string()]);
    }

    #[test]
    fn test_dataframe_add_column_preserves_index() {
        let mut df = setup_test_dataframe();
        let original_index = df.index.clone();

        df.add_column(
            "NewColumn".to_string(),
            vec![
                Some(DataFrameValue::Integer(10)),
                Some(DataFrameValue::Integer(20)),
                Some(DataFrameValue::Integer(30)),
            ],
        )
        .unwrap();

        assert_eq!(df.index, original_index);
    }

    #[test]
    fn test_loc_functionality() {
        // Setup Test DataFrame
        let df = setup_test_dataframe();

        // Test Valid Inputs
        let result = df.loc(&[1, 2], Some(&["Name", "Score"])).unwrap();
        assert_eq!(result.columns.len(), 2, "Expected 2 columns in the result");
        assert_eq!(result.data.len(), 2, "Expected data for 2 rows");

        // Test Invalid Row Indices
        assert!(
            df.loc(&[999], Some(&["Name"])).is_err(),
            "Expected error for out-of-bounds row index"
        );
    }

    #[test]
    fn test_iloc_valid_access() {
        let df = setup_test_dataframe();

        // Test iloc for a valid row and column
        let value = df.iloc(1, 1).unwrap();
        assert_eq!(*value, Some(DataFrameValue::String("Bob".to_string())));
    }

    #[test]
    fn test_iloc_row_out_of_bounds() {
        let df = setup_test_dataframe();

        // Test iloc with a row index that is out of bounds
        let result = df.iloc(99, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_iloc_column_out_of_bounds() {
        let df = setup_test_dataframe();

        // Test iloc with a column index that is out of bounds
        let result = df.iloc(1, 99);
        assert!(result.is_err());
    }

    #[test]
    fn test_boolean_index() {
        let df = setup_test_dataframe();

        // Create a boolean vector where only the second row (Bob) is true
        let condition = vec![false, true, false];

        let filtered_df = df.boolean_index(condition).unwrap();

        // Check that the filtered DataFrame only contains the second row
        assert_eq!(
            filtered_df.data.get("Name").unwrap()[0],
            Some(DataFrameValue::String("Bob".to_string()))
        );
        assert_eq!(
            filtered_df.data.get("Score").unwrap()[0],
            Some(DataFrameValue::Float(4.0))
        );

        // Check that the DataFrame has only one row
        assert_eq!(filtered_df.data.get("ID").unwrap().len(), 1);

        // Check that the index is correctly updated to only include the second row
        assert_eq!(filtered_df.index, vec!["1".to_string()]);
    }

    #[test]
    fn test_filter_by_condition() {
        let df = setup_test_dataframe();

        let condition_results = df
            .apply_condition("Score", |value| match value {
                DataFrameValue::Float(score) => *score > 3.0,
                _ => false,
            })
            .expect("Failed to apply condition");

        let filtered_df = df.filter_by_condition(condition_results);

        // checking and asserting
        assert_eq!(filtered_df.data["ID"].len(), 2);
        assert_eq!(
            filtered_df.data["Name"][0],
            Some(DataFrameValue::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_save_and_read_dataframe() {
        let df_original = setup_test_dataframe();
        let file_path = "test_dataframe.bin"; //  binary format for serialization

        // Save original DataFrame to file
        df_original.save_to_file(file_path).unwrap();

        // Load DataFrame from file
        let df_loaded = DataFrame::read_from_file(file_path).unwrap();

        // Compare columns (order does not matter)
        assert_eq!(df_original.columns.len(), df_loaded.columns.len());
        assert!(df_original
            .columns
            .iter()
            .all(|col| df_loaded.columns.contains(col)));

        // Compare index (order should matter here)
        assert_eq!(df_original.index, df_loaded.index);

        // Compare data for each column (order does not matter for the columns, but matters for the values within them)
        for column_name in df_original.columns.iter() {
            let original_values = df_original.data.get(column_name).unwrap();
            let loaded_values = df_loaded.data.get(column_name).unwrap();
            assert_eq!(
                original_values, loaded_values,
                "Column values do not match for {}",
                column_name
            );
        }

        // Clean up test file
        std::fs::remove_file(file_path).unwrap();
    }

    /// Tests the `dropna()` method to ensure it correctly drops rows containing `None` values.
    #[test]
    fn test_dropna() {
        // Initialize column data as a HashMap
        let mut column_data = HashMap::new();
        column_data.insert(
            "ID".to_string(),
            vec![Some(DataFrameValue::Integer(1)), None],
        );
        column_data.insert(
            "Name".to_string(),
            vec![
                Some(DataFrameValue::String("Alice".to_string())),
                Some(DataFrameValue::String("Bob".to_string())),
            ],
        );
        column_data.insert(
            "Score".to_string(),
            vec![
                Some(DataFrameValue::Float(3.5)),
                Some(DataFrameValue::Float(4.0)),
            ],
        );

        // Column names vector remains the same
        let columns = vec!["ID".to_string(), "Name".to_string(), "Score".to_string()];

        let mut df = DataFrame::new(column_data, columns).unwrap();

        df.dropna();
        assert!(
            df.data.get("ID").unwrap().len() == 1,
            "ID column should only contain 1 value after dropna"
        );
        assert_eq!(
            df.data.get("Name").unwrap()[0],
            Some(DataFrameValue::String("Alice".to_string())),
            "Name column should contain 'Alice'"
        );
        assert_eq!(
            df.data.get("Score").unwrap()[0],
            Some(DataFrameValue::Float(3.5)),
            "Score column should contain 3.5 as the first value"
        );
    }

    #[test]
    fn test_fillna() {
        // Initialize column data as a HashMap
        let mut column_data = HashMap::new();
        column_data.insert(
            "ID".to_string(),
            vec![Some(DataFrameValue::Integer(1)), None],
        );
        column_data.insert(
            "Name".to_string(),
            vec![Some(DataFrameValue::String("Alice".to_string())), None],
        );
        column_data.insert(
            "Score".to_string(),
            vec![Some(DataFrameValue::Float(3.5)), None],
        );

        // Column names vector
        let columns = vec!["ID".to_string(), "Name".to_string(), "Score".to_string()];

        // Initialize the DataFrame
        let mut df = DataFrame::new(column_data, columns).unwrap();

        df.fillna("Score", DataFrameValue::Float(0.0)); // Example: Fill `None` in "Score" column with 0.0

        // Assert that the None values are replaced as expected
        assert_eq!(
            df.data.get("ID").unwrap()[1],
            None,
            "ID column should still contain None for second value"
        );
        assert_eq!(
            df.data.get("Name").unwrap()[1],
            None,
            "Name column should still contain None for second value"
        );
        assert_eq!(
            df.data.get("Score").unwrap()[1],
            Some(DataFrameValue::Float(0.0)),
            "Score column second value should be filled with 0.0"
        );
    }

    #[test]
    fn test_query() {
        let df = setup_test_dataframe();
        let result = df.query("ID == 1");
        assert_eq!(result.index.len(), 1);
        assert_eq!(result.index[0], "0");
        // Check the Name field
        let name = result.data.get("Name").unwrap().get(0).unwrap();
        assert_eq!(name, &Some(DataFrameValue::String("Alice".to_string())));
    }
}
