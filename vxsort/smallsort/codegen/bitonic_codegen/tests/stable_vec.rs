use std::sync::Arc;

use bitonic_codegen::stable_vec::StableVec;

#[test]
fn stable_vec_keeps_published_entries_stable_across_chunk_growth() {
    let values = Arc::new(StableVec::<String, 1, 8>::new());
    let first_index = values.push("first".to_owned());
    let first_address = values.get(first_index) as *const String;

    for index in 0..32 {
        values.push(format!("value-{index}"));
    }

    assert_eq!(values.get(first_index) as *const String, first_address);
    assert_eq!(values.get(first_index), "first");
}

#[test]
fn stable_vec_snapshot_bounds_iteration_to_capture_time_len() {
    let values = Arc::new(StableVec::<usize, 1, 8>::new());
    values.push(1);
    values.push(2);
    let snapshot = values.snapshot();

    values.push(3);

    assert_eq!(snapshot.iter().copied().collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(
        values.snapshot().iter().copied().collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}
