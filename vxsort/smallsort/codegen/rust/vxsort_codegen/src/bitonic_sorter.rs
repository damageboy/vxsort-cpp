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
    stages: Vec<BitonicStage>,
}

impl BitonicSorter {
    pub fn new(n: usize) -> Self {
        let mut sorter = Self { stages: Vec::new() };
        sorter.generate_bitonic_sorter(n, 0, 0);
        sorter
    }

    pub fn stages(&self) -> &[BitonicStage] {
        &self.stages
    }

    fn generate_bitonic_sorter(&mut self, n: usize, stage: usize, start: usize) -> usize {
        if n == 1 {
            return stage;
        }

        let half = n / 2;
        self.generate_bitonic_sorter(half, stage, start);
        let stage = self.generate_bitonic_sorter(half, stage, start + half);
        self.generate_bitonic_merge(n, stage, start, true)
    }

    fn generate_bitonic_merge(
        &mut self,
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

        self.add_ops(stage, pairs);
        self.generate_bitonic_merge(half, stage + 1, start, false);
        self.generate_bitonic_merge(half, stage + 1, start + half, false)
    }

    fn add_ops(&mut self, stage: usize, pairs: Vec<(usize, usize)>) {
        while self.stages.len() <= stage {
            let index = self.stages.len();
            self.stages.push(BitonicStage::new(index, Vec::new()));
        }
        self.stages[stage].pairs.extend(pairs);
    }
}
