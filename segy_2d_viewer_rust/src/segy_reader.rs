// SEG-Y Reader Module - mmap + rayon based
use anyhow::{Context, Result};
use byteorder::{BigEndian, ByteOrder};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

pub struct SegyReader {
    pub data: Vec<Vec<f32>>,           // [samples][traces]
    pub raw_data: Vec<Vec<f32>>,       // raw data before normalization
    pub num_traces: usize,
    pub num_samples: usize,
    pub sample_rate: f32,
}

impl SegyReader {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            raw_data: Vec::new(),
            num_traces: 0,
            num_samples: 0,
            sample_rate: 0.0,
        }
    }

    pub fn load_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let file = File::open(&path).context("Failed to open SEG-Y file")?;
        let mmap = unsafe { Mmap::map(&file).context("Failed to mmap file")? };

        // Read binary header
        if mmap.len() < 3600 {
            anyhow::bail!("File too small to be a valid SEG-Y file");
        }

        // Read number of samples (bytes 3220-3221, 2 bytes, big-endian)
        self.num_samples = BigEndian::read_u16(&mmap[3220..3222]) as usize;

        // Read sample interval (bytes 3216-3217, microseconds)
        let sample_interval_us = BigEndian::read_u16(&mmap[3216..3218]) as f32;
        self.sample_rate = sample_interval_us / 1_000_000.0; // to seconds

        // Calculate number of traces
        let header_size = 3600;
        let trace_size = 240 + self.num_samples * 4;
        self.num_traces = (mmap.len() - header_size) / trace_size;

        println!(
            "Loading SEG-Y: {} traces, {} samples",
            self.num_traces, self.num_samples
        );

        // Parallel load traces
        let trace_indices: Vec<usize> = (0..self.num_traces).collect();
        let traces: Vec<Vec<f32>> = trace_indices
            .par_iter()
            .map(|&trace_idx| {
                Self::load_trace(&mmap, trace_idx, self.num_samples, header_size, trace_size)
            })
            .collect();

        // Transpose: Vec<Vec<f32>> where outer is traces -> [samples][traces]
        self.raw_data = vec![vec![0.0; self.num_traces]; self.num_samples];
        for (trace_idx, trace) in traces.iter().enumerate() {
            for (sample_idx, &value) in trace.iter().enumerate() {
                self.raw_data[sample_idx][trace_idx] = value;
            }
        }

        // Normalize data
        self.data = self.normalize_data(&self.raw_data);

        Ok(())
    }

    fn load_trace(
        mmap: &Mmap,
        trace_idx: usize,
        num_samples: usize,
        header_size: usize,
        trace_size: usize,
    ) -> Vec<f32> {
        let mut trace_data = vec![0.0; num_samples];
        let trace_offset = header_size + trace_idx * trace_size + 240; // skip trace header

        for i in 0..num_samples {
            let offset = trace_offset + i * 4;
            if offset + 4 > mmap.len() {
                break;
            }

            let ibm_bytes = &mmap[offset..offset + 4];
            trace_data[i] = Self::ibm_to_ieee(ibm_bytes);
        }

        trace_data
    }

    fn ibm_to_ieee(bytes: &[u8]) -> f32 {
        let ibm_int = BigEndian::read_u32(bytes);

        let sign = (ibm_int >> 31) & 1;
        let exponent = ((ibm_int >> 24) & 0x7f) as i32;
        let mantissa = ibm_int & 0x00ffffff;

        if mantissa == 0 {
            return 0.0;
        }

        // Convert mantissa to [0.0625, 1.0)
        let mut value = mantissa as f64 / 16777216.0; // 2^24

        // Calculate exponent (overflow protection)
        let exp_val = exponent - 64;
        if exp_val > 60 || exp_val < -60 {
            return 0.0;
        }

        value *= 16.0_f64.powi(exp_val);

        if sign == 1 {
            value = -value;
        }

        // Check for inf/nan
        if value.is_finite() {
            value as f32
        } else {
            0.0
        }
    }

    fn normalize_data(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut normalized = data.to_vec();

        // Normalize each trace (column)
        for trace_idx in 0..self.num_traces {
            let mut max_abs = 0.0f32;

            // Find max absolute value in this trace
            for sample_idx in 0..self.num_samples {
                let val = data[sample_idx][trace_idx].abs();
                if val.is_finite() && val > max_abs {
                    max_abs = val;
                }
            }

            // Normalize
            if max_abs > 0.0 {
                for sample_idx in 0..self.num_samples {
                    let val = data[sample_idx][trace_idx];
                    normalized[sample_idx][trace_idx] = if val.is_finite() {
                        val / max_abs
                    } else {
                        0.0
                    };
                }
            } else {
                for sample_idx in 0..self.num_samples {
                    normalized[sample_idx][trace_idx] = 0.0;
                }
            }
        }

        normalized
    }

    pub fn get_trace(&self, trace_idx: usize) -> Option<Vec<f32>> {
        if trace_idx >= self.num_traces {
            return None;
        }

        let mut trace = vec![0.0; self.num_samples];
        for sample_idx in 0..self.num_samples {
            trace[sample_idx] = self.raw_data[sample_idx][trace_idx];
        }
        Some(trace)
    }
}
