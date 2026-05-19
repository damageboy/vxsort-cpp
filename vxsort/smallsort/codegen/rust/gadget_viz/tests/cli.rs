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
fn cli_writes_html_for_selected_template_record() {
    let dir = std::env::temp_dir().join(format!("gadget_viz_cli_{}_html", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    let input = dir.join("records.jsonl");
    let output = dir.join("record.html");
    fs::write(
        &input,
        r#"{"kind":"template","arch":"avx2","dtype":"i64","gadget_depth":1,"tier":"shallow","graph":{"top":{"kind":"intrinsic","name":"_mm256_permute_pd","isomorphic_order":true,"operands":{"a":{"kind":"input","name":"top"},"imm8":{"kind":"symbolic","role":"imm8","bits":8,"var_id":"v0"}}},"bottom":null}}
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
    assert!(html.contains("top input"));
}

#[test]
fn cli_rejects_synthesized_records() {
    let dir = std::env::temp_dir().join(format!(
        "gadget_viz_cli_{}_reject_synthesized",
        std::process::id()
    ));
    fs::create_dir_all(&dir).expect("temp dir");
    let input = dir.join("records.jsonl");
    let output = dir.join("record.html");
    fs::write(
        &input,
        r#"{"kind":"synthesized"}
"#,
    )
    .expect("write input");

    let output_result = Command::new(gadget_viz_bin())
        .args([
            "--input",
            input.to_str().unwrap(),
            "--format",
            "html",
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("gadget_viz should run");

    assert!(!output_result.status.success());
    let stderr = String::from_utf8_lossy(&output_result.stderr);
    assert!(stderr.contains("only renders template records"), "{stderr}");
    assert!(!output.exists());
}
