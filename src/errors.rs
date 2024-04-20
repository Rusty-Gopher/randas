// src/error.rs
use rusqlite;
use std::fmt;

// Here is a custom error type that can hold different kinds of errors
#[derive(Debug)]
pub enum MyError {
    Rusqlite(rusqlite::Error),
    DataFrame(String),
}

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MyError::Rusqlite(err) => write!(f, "SQLite error: {}", err),
            MyError::DataFrame(err) => write!(f, "DataFrame error: {}", err),
        }
    }
}

impl std::error::Error for MyError {}

impl From<rusqlite::Error> for MyError {
    fn from(err: rusqlite::Error) -> Self {
        MyError::Rusqlite(err)
    }
}
