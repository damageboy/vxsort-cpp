#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BitonicStage {
    index: usize,
    pairs: Vec<(usize, usize)>,
}

impl BitonicStage {
    pub fn new(index: usize, pairs: Vec<(usize, usize)>) -> Self {
        Self { index, pairs }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BitonicSorter {
    stages: Box<[BitonicStage]>,
}

impl BitonicSorter {
    pub fn new(n: usize) -> Self {
        let mut stages = Vec::new();
        Self::generate_bitonic_sorter(&mut stages, n, 0, 0);
        Self {
            stages: stages.into_boxed_slice(),
        }
    }

    pub fn stages(&self) -> &[BitonicStage] {
        &self.stages
    }

    fn generate_bitonic_sorter(
        stages: &mut Vec<BitonicStage>,
        n: usize,
        stage: usize,
        start: usize,
    ) -> usize {
        if n == 1 {
            return stage;
        }

        let half = n / 2;
        Self::generate_bitonic_sorter(stages, half, stage, start);
        let stage = Self::generate_bitonic_sorter(stages, half, stage, start + half);
        Self::generate_bitonic_merge(stages, n, stage, start, true)
    }

    fn generate_bitonic_merge(
        stages: &mut Vec<BitonicStage>,
        n: usize,
        stage: usize,
        start: usize,
        initial_merge: bool,
    ) -> usize {
        if n == 1 {
            return stage;
        }

        let half = n / 2;
        let pairs = if initial_merge {
            (start + 1..=start + half)
                .zip((start + half + 1..=start + n).rev())
                .collect()
        } else {
            (start + 1..=start + half)
                .map(|left| (left, left + half))
                .collect()
        };

        Self::add_ops(stages, stage, pairs);
        Self::generate_bitonic_merge(stages, half, stage + 1, start, false);
        Self::generate_bitonic_merge(stages, half, stage + 1, start + half, false)
    }

    fn add_ops(stages: &mut Vec<BitonicStage>, stage: usize, pairs: Vec<(usize, usize)>) {
        while stages.len() <= stage {
            let index = stages.len();
            stages.push(BitonicStage::new(index, Vec::new()));
        }
        stages[stage].pairs.extend(pairs);
    }
}
