use std::fs;
use std::process::Command;

fn gadget_viz_bin() -> String {
    std::env::var("CARGO_BIN_EXE_gadget_viz").unwrap_or_else(|_| {
        let mut path = std::env::current_exe().expect("test executable path");
        path.pop();
        if path.ends_with("deps") {
            path.pop();
        }
        path.push("gadget_viz");
        path.to_string_lossy().into_owned()
    })
}

#[test]
fn cli_writes_html_for_selected_record() {
    let dir = std::env::temp_dir().join(format!("gadget_viz_cli_{}_html", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    let input = dir.join("records.jsonl");
    let output = dir.join("record.html");
    fs::write(
        &input,
        r#"{"kind":"synthesized","arch":"avx2","dtype":"i64","gadget_depth":1,"fixture":"identity_pairs","input_state":{"top":[0,1,2,3],"bottom":[4,5,6,7]},"target_pairs":[[0,4],[1,5],[2,6],[3,7]],"output_state":{"top":[0,1,2,3],"bottom":[4,5,6,7]},"top_instructions":[{"name":"_mm256_permute_pd","args":{"a":"top","imm8":0}}],"bottom_instructions":[]}
"#,
    )
    .expect("write input");

    let status = Command::new(gadget_viz_bin())
        .args([
            "--input",
            input.to_str().unwrap(),
            "--index",
            "0",
            "--format",
            "html",
            "--output",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("gadget_viz should run");
    assert!(status.success());

    let html = fs::read_to_string(output).expect("html output");
    assert!(html.contains("mermaid"));
    assert!(html.contains("_mm256_permute_pd"));
    assert!(html.contains("[0,1,2,3]"));
}
