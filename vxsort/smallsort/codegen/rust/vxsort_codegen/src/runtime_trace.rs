use std::{
    fs::File,
    io::{LineWriter, Write},
    path::Path,
    time::Instant,
};

use serde_json::{Map, Value, json};

/// Emits a runtime trace event only when tracing is enabled.
///
/// The macro deliberately guards the `json!` construction so expensive field
/// expressions are not evaluated for normal runs without a runtime trace.
///
/// Use the guarded form when a call site has an additional sampling/detail
/// condition:
///
/// ```ignore
/// trace!(trace, "wave_started", { "wave": wave });
/// trace!(trace, if should_sample(order), "scoring_job", { "order": order });
/// ```
#[macro_export]
macro_rules! trace {
    ($trace:expr, if $guard:expr, $event:literal, [ $($prep:stmt;)* ], { $($fields:tt)* }) => {{
        if ($guard) && $trace.is_enabled() {
            $($prep)*
            $trace.event($event, ::serde_json::json!({ $($fields)* }));
        }
    }};
    ($trace:expr, $event:literal, [ $($prep:stmt;)* ], { $($fields:tt)* }) => {{
        if $trace.is_enabled() {
            $($prep)*
            $trace.event($event, ::serde_json::json!({ $($fields)* }));
        }
    }};
    ($trace:expr, if $guard:expr, $event:literal, { $($fields:tt)* }) => {{
        if ($guard) && $trace.is_enabled() {
            $trace.event($event, ::serde_json::json!({ $($fields)* }));
        }
    }};
    ($trace:expr, $event:literal, { $($fields:tt)* }) => {{
        if $trace.is_enabled() {
            $trace.event($event, ::serde_json::json!({ $($fields)* }));
        }
    }};
}

pub struct RuntimeTrace {
    writer: Option<LineWriter<File>>,
    started_at: Instant,
    next_sequence: u64,
    write_error: Option<String>,
}

impl RuntimeTrace {
    pub fn disabled() -> Self {
        Self {
            writer: None,
            started_at: Instant::now(),
            next_sequence: 0,
            write_error: None,
        }
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::create(path).map_err(|error| {
            format!("failed to create runtime trace {}: {error}", path.display())
        })?;
        Ok(Self {
            writer: Some(LineWriter::new(file)),
            started_at: Instant::now(),
            next_sequence: 0,
            write_error: None,
        })
    }

    pub fn from_path(path: Option<&Path>) -> Result<Self, String> {
        match path {
            Some(path) => Self::open(path),
            None => Ok(Self::disabled()),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.writer.is_some() && self.write_error.is_none()
    }

    pub fn event(&mut self, event: &str, fields: Value) {
        if !self.is_enabled() {
            return;
        }

        let mut payload = match fields {
            Value::Object(fields) => fields,
            other => {
                let mut fields = Map::new();
                fields.insert("fields".to_owned(), other);
                fields
            }
        };
        payload.insert("event".to_owned(), json!(event));
        payload.insert("sequence".to_owned(), json!(self.next_sequence));
        payload.insert("elapsed_us".to_owned(), json!(self.elapsed_us()));
        self.next_sequence += 1;

        let Some(writer) = self.writer.as_mut() else {
            return;
        };

        if let Err(error) = serde_json::to_writer(&mut *writer, &Value::Object(payload)) {
            self.write_error = Some(format!("failed to write runtime trace event: {error}"));
            return;
        }
        if let Err(error) = writer.write_all(b"\n") {
            self.write_error = Some(format!("failed to write runtime trace newline: {error}"));
        }
    }

    pub fn flush(&mut self) -> Result<(), String> {
        if let Some(error) = &self.write_error {
            return Err(error.clone());
        }
        if let Some(writer) = self.writer.as_mut() {
            writer
                .flush()
                .map_err(|error| format!("failed to flush runtime trace: {error}"))?;
        }
        Ok(())
    }

    fn elapsed_us(&self) -> u64 {
        self.started_at
            .elapsed()
            .as_micros()
            .min(u128::from(u64::MAX)) as u64
    }
}

impl Default for RuntimeTrace {
    fn default() -> Self {
        Self::disabled()
    }
}

impl Drop for RuntimeTrace {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
