// Auto Picking Algorithms Module
use rayon::prelude::*;

pub struct AutoPicker;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    StaLta,
    EnergyRatio,
    Aic,
}

impl AutoPicker {
    pub fn pick_all_traces(
        data: &[Vec<f32>],
        num_traces: usize,
        num_samples: usize,
        algorithm: Algorithm,
    ) -> Vec<(usize, f32)> {
        let trace_indices: Vec<usize> = (0..num_traces).collect();

        trace_indices
            .par_iter()
            .filter_map(|&trace_idx| {
                // Extract trace
                let mut trace = vec![0.0; num_samples];
                for sample_idx in 0..num_samples {
                    trace[sample_idx] = data[sample_idx][trace_idx];
                }

                // Pick
                let sample_idx = match algorithm {
                    Algorithm::StaLta => Self::pick_sta_lta(&trace, 10, 40, 1.5),
                    Algorithm::EnergyRatio => Self::pick_energy_ratio(&trace, 20, 0.05),
                    Algorithm::Aic => Self::pick_aic(&trace, 200),
                };

                sample_idx.map(|s| (trace_idx, s))
            })
            .collect()
    }

    fn pick_sta_lta(trace: &[f32], sta_window: usize, lta_window: usize, threshold: f32) -> Option<f32> {
        let len = trace.len();
        if len < lta_window + sta_window {
            return None;
        }

        let trace_abs: Vec<f32> = trace.iter().map(|v| v.abs()).collect();

        let mut max_ratio = 0.0f32;
        let mut best_idx = 0;
        let mut found = false;

        for i in lta_window..(len - sta_window) {
            let lta: f32 = trace_abs[i - lta_window..i].iter().sum::<f32>() / lta_window as f32;
            let sta: f32 = trace_abs[i..i + sta_window].iter().sum::<f32>() / sta_window as f32;

            if lta > 0.0 {
                let ratio = sta / lta;
                if ratio > threshold && ratio > max_ratio {
                    max_ratio = ratio;
                    best_idx = i;
                    found = true;
                }
            }
        }

        if found {
            Some(best_idx as f32)
        } else {
            // Fallback: find the point with maximum absolute amplitude
            let max_idx = trace_abs.iter()
                .enumerate()
                .skip(lta_window)
                .take(len - lta_window - sta_window)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)?;
            Some(max_idx as f32)
        }
    }

    fn pick_energy_ratio(trace: &[f32], window: usize, threshold: f32) -> Option<f32> {
        let len = trace.len();
        if len < window {
            return None;
        }

        let energy: Vec<f32> = trace.iter().map(|v| v * v).collect();
        let total_energy: f32 = energy.iter().sum();

        if total_energy == 0.0 {
            return None;
        }

        let mut cumulative = 0.0;
        for (i, &e) in energy.iter().enumerate() {
            cumulative += e;
            let ratio = cumulative / total_energy;
            if ratio > threshold {
                return Some(i as f32);
            }
        }

        None
    }

    fn pick_aic(trace: &[f32], window: usize) -> Option<f32> {
        let len = trace.len();
        if len < window {
            return None;
        }

        let start_idx = (len as f32 * 0.05) as usize;
        let end_idx = window.min(len);

        let mut min_aic = f32::INFINITY;
        let mut min_idx = start_idx;

        for k in start_idx..end_idx {
            if k < 2 || k > len - 2 {
                continue;
            }

            // Variance of two segments
            let var1 = Self::variance(&trace[..k]);
            let var2 = Self::variance(&trace[k..window.min(len)]);

            if var1 > 0.0 && var2 > 0.0 {
                let aic = (k as f32) * var1.ln() + ((len - k) as f32) * var2.ln();
                if aic < min_aic {
                    min_aic = aic;
                    min_idx = k;
                }
            }
        }

        if min_aic < f32::INFINITY {
            Some(min_idx as f32)
        } else {
            None
        }
    }

    fn variance(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
        var
    }
}
