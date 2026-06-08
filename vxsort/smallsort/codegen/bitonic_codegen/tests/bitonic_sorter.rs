use bitonic_codegen::bitonic_sorter::BitonicSorter;

#[test]
fn generates_python_stage_order_for_eight_elements() {
    let sorter = BitonicSorter::new(8);

    let stages = sorter
        .stages()
        .iter()
        .map(|stage| stage.pairs().to_vec())
        .collect::<Vec<_>>();

    assert_eq!(
        stages,
        vec![
            vec![(1, 2), (3, 4), (5, 6), (7, 8)],
            vec![(1, 4), (2, 3), (5, 8), (6, 7)],
            vec![(1, 2), (3, 4), (5, 6), (7, 8)],
            vec![(1, 8), (2, 7), (3, 6), (4, 5)],
            vec![(1, 3), (2, 4), (5, 7), (6, 8)],
            vec![(1, 2), (3, 4), (5, 6), (7, 8)],
        ]
    );
}

#[test]
fn exposes_stage_indices_with_pairs() {
    let sorter = BitonicSorter::new(8);

    assert_eq!(sorter.stages()[0].index(), 0);
    assert_eq!(sorter.stages()[3].index(), 3);
    assert_eq!(
        sorter.stages()[3].pairs(),
        &[(1, 8), (2, 7), (3, 6), (4, 5)]
    );
}
