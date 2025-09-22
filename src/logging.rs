use std::io::{self, Write, Read, Seek, SeekFrom};
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::sync::Mutex;
use tracing::Level;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tracing_subscriber::Layer;
use tracing_subscriber::filter::LevelFilter;

pub fn init_logging(log_level: Level, log_file: Option<&str>) {
    let level_filter = LevelFilter::from_level(log_level);
    let stdout_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stdout);

    if let Some(path) = log_file {
        let capped_writer = make_capped_file_writer(PathBuf::from(path), 10 * 1024 * 1024);
        let file_layer = tracing_subscriber::fmt::layer().with_writer(capped_writer);
        tracing_subscriber::registry()
            .with(stdout_layer.with_filter(level_filter))
            .with(file_layer.with_filter(level_filter))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(stdout_layer.with_filter(level_filter))
            .init();
    }
}

fn make_capped_file_writer(path: PathBuf, max_len: u64) -> impl Fn() -> CappedFileWriter {
    let lock = std::sync::Arc::new(Mutex::new(()));
    move || CappedFileWriter { path: path.clone(), max_len, lock: lock.clone() }
}

struct CappedFileWriter {
    path: PathBuf,
    max_len: u64,
    lock: std::sync::Arc<Mutex<()>>,
}

impl Write for CappedFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let _guard = self.lock.lock().unwrap();

        let mut need_truncate = false;
        if let Ok(meta) = std::fs::metadata(&self.path) {
            if meta.len() >= self.max_len { need_truncate = true; }
        }

        if need_truncate {
            let keep_bytes = self.max_len / 2;
            let mut tail = Vec::new();
            if let Ok(mut rf) = OpenOptions::new().read(true).open(&self.path) {
                if let Ok(meta) = rf.metadata() {
                    let size = meta.len();
                    let start = if size > keep_bytes { size - keep_bytes } else { 0 };
                    if rf.seek(SeekFrom::Start(start)).is_ok() {
                        let _ = rf.read_to_end(&mut tail);
                    }
                }
            }
            let mut wf = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&self.path)?;
            if !tail.is_empty() {
                wf.write_all(&tail)?;
            }
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
