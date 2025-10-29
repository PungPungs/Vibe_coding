// Picking Manager Module
use std::collections::HashMap;

pub struct PickingManager {
    picks: HashMap<usize, f32>, // trace_idx -> sample_idx
    interpolated: Vec<f32>,     // interpolated picks for all traces
    num_traces: usize,
    enabled: bool,
}

impl PickingManager {
    pub fn new() -> Self {
        Self {
            picks: HashMap::new(),
            interpolated: Vec::new(),
            num_traces: 0,
            enabled: true,
        }
    }

    pub fn set_num_traces(&mut self, num_traces: usize) {
        self.num_traces = num_traces;
        self.interpolated = vec![-1.0; num_traces];
    }

    pub fn add_pick(&mut self, trace_idx: usize, sample_idx: f32) {
        if !self.enabled || trace_idx >= self.num_traces {
            return;
        }

        self.picks.insert(trace_idx, sample_idx);
        self.interpolate();
    }

    pub fn clear_picks(&mut self) {
        self.picks.clear();
        self.interpolated.fill(-1.0);
    }

    pub fn set_picks(&mut self, picks: Vec<(usize, f32)>) {
        self.picks.clear();
        for (trace_idx, sample_idx) in picks {
            self.picks.insert(trace_idx, sample_idx);
        }
        self.interpolate();
    }

    pub fn get_picks(&self) -> Vec<(usize, f32)> {
        let mut picks: Vec<_> = self.picks.iter().map(|(&t, &s)| (t, s)).collect();
        picks.sort_by_key(|(t, _)| *t);
        picks
    }

    pub fn get_interpolated(&self) -> &[f32] {
        &self.interpolated
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn interpolate(&mut self) {
        self.interpolated.fill(-1.0);

        if self.picks.is_empty() {
            return;
        }

        let mut sorted_picks: Vec<_> = self.picks.iter().collect();
        sorted_picks.sort_by_key(|(t, _)| *t);

        if sorted_picks.len() == 1 {
            let (&trace_idx, &sample_idx) = sorted_picks[0];
            self.interpolated[trace_idx] = sample_idx;
            return;
        }

        // Linear interpolation
        for i in 0..sorted_picks.len() - 1 {
            let (&trace1, &sample1) = sorted_picks[i];
            let (&trace2, &sample2) = sorted_picks[i + 1];

            for trace_idx in trace1..=trace2 {
                if trace2 != trace1 {
                    let t = (trace_idx - trace1) as f32 / (trace2 - trace1) as f32;
                    self.interpolated[trace_idx] = sample1 + t * (sample2 - sample1);
                } else {
                    self.interpolated[trace_idx] = sample1;
                }
            }
        }
    }

    pub fn save_to_file(&self, filename: &str) -> anyhow::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename)?;
        writeln!(file, "Trace,Sample")?;

        for (trace_idx, sample_idx) in self.get_picks() {
            writeln!(file, "{},{}", trace_idx, sample_idx)?;
        }

        Ok(())
    }

    pub fn load_from_file(&mut self, filename: &str) -> anyhow::Result<()> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        self.picks.clear();

        for (idx, line) in reader.lines().enumerate() {
            if idx == 0 {
                continue; // Skip header
            }

            let line = line?;
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                if let (Ok(trace_idx), Ok(sample_idx)) =
                    (parts[0].parse::<usize>(), parts[1].parse::<f32>())
                {
                    self.picks.insert(trace_idx, sample_idx);
                }
            }
        }

        self.interpolate();
        Ok(())
    }
}
