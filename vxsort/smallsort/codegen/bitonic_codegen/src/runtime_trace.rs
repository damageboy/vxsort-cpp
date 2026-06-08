use std::{
    fs::File,
    io::{LineWriter, Write},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
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

#[derive(Clone)]
pub struct RuntimeTrace {
    inner: Arc<RuntimeTraceInner>,
}

struct RuntimeTraceInner {
    writer: Mutex<Option<LineWriter<File>>>,
    started_at: Instant,
    next_sequence: AtomicU64,
    write_error: Mutex<Option<String>>,
}

impl RuntimeTrace {
    pub fn disabled() -> Self {
        Self {
            inner: Arc::new(RuntimeTraceInner {
                writer: Mutex::new(None),
                started_at: Instant::now(),
                next_sequence: AtomicU64::new(0),
                write_error: Mutex::new(None),
            }),
        }
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::create(path).map_err(|error| {
            format!("failed to create runtime trace {}: {error}", path.display())
        })?;
        Ok(Self {
            inner: Arc::new(RuntimeTraceInner {
                writer: Mutex::new(Some(LineWriter::new(file))),
                started_at: Instant::now(),
                next_sequence: AtomicU64::new(0),
                write_error: Mutex::new(None),
            }),
        })
    }

    pub fn from_path(path: Option<&Path>) -> Result<Self, String> {
        match path {
            Some(path) => Self::open(path),
            None => Ok(Self::disabled()),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.inner
            .writer
            .lock()
            .expect("runtime trace writer lock should not be poisoned")
            .is_some()
            && self
                .inner
                .write_error
                .lock()
                .expect("runtime trace error lock should not be poisoned")
                .is_none()
    }

    pub fn event(&self, event: &str, fields: Value) {
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
        let sequence = self.inner.next_sequence.fetch_add(1, Ordering::Relaxed);
        payload.insert("sequence".to_owned(), json!(sequence));
        payload.insert("elapsed_us".to_owned(), json!(self.elapsed_us()));

        let mut writer = self
            .inner
            .writer
            .lock()
            .expect("runtime trace writer lock should not be poisoned");
        let Some(writer) = writer.as_mut() else {
            return;
        };

        if let Err(error) = serde_json::to_writer(&mut *writer, &Value::Object(payload)) {
            *self
                .inner
                .write_error
                .lock()
                .expect("runtime trace error lock should not be poisoned") =
                Some(format!("failed to write runtime trace event: {error}"));
            return;
        }
        if let Err(error) = writer.write_all(b"\n") {
            *self
                .inner
                .write_error
                .lock()
                .expect("runtime trace error lock should not be poisoned") =
                Some(format!("failed to write runtime trace newline: {error}"));
        }
    }

    pub fn flush(&self) -> Result<(), String> {
        if let Some(error) = &*self
            .inner
            .write_error
            .lock()
            .expect("runtime trace error lock should not be poisoned")
        {
            return Err(error.clone());
        }
        if let Some(writer) = self
            .inner
            .writer
            .lock()
            .expect("runtime trace writer lock should not be poisoned")
            .as_mut()
        {
            writer
                .flush()
                .map_err(|error| format!("failed to flush runtime trace: {error}"))?;
        }
        Ok(())
    }

    fn elapsed_us(&self) -> u64 {
        self.inner
            .started_at
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
