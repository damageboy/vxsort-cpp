use std::fs;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use gadget_viz::{OutputFormat, read_jsonl, render_records, select_records};

#[derive(Parser, Debug)]
#[command(about = "Render gadget synthesizer JSONL records as Mermaid/Markdown/HTML")]
struct Args {
    /// Input JSONL dump from dump_gadget_synth.py or gadget_synth dump_gadget_synth.
    #[arg(long)]
    input: PathBuf,

    /// Optional zero-based record index. If omitted, render all records.
    #[arg(long)]
    index: Option<usize>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = FormatArg::Html)]
    format: FormatArg,

    /// Output path. If omitted, write to stdout.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Document title for markdown/html output.
    #[arg(long)]
    title: Option<String>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FormatArg {
    Mermaid,
    Markdown,
    Html,
}

impl From<FormatArg> for OutputFormat {
    fn from(value: FormatArg) -> Self {
        match value {
            FormatArg::Mermaid => Self::Mermaid,
            FormatArg::Markdown => Self::Markdown,
            FormatArg::Html => Self::Html,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let records = read_jsonl(&args.input)?;
    let selected = select_records(&records, args.index)?;
    let title = args.title.unwrap_or_else(|| {
        if let Some(index) = args.index {
            format!("{} record {index}", args.input.display())
        } else {
            format!("{} ({} records)", args.input.display(), selected.len())
        }
    });
    let rendered = render_records(&selected, args.format.into(), &title)?;

    if let Some(output) = args.output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(output, rendered)?;
    } else {
        print!("{rendered}");
    }

    Ok(())
}
