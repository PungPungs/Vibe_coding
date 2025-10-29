// Main Application with eframe + egui + OpenGL
mod auto_picking;
mod gl_renderer;
mod picking_manager;
mod segy_reader;

use auto_picking::{Algorithm, AutoPicker};
use eframe::egui;
use gl_renderer::GlRenderer;
use glow::HasContext;
use parking_lot::Mutex;
use picking_manager::PickingManager;
use segy_reader::SegyReader;
use std::env;
use std::sync::Arc;

fn main() -> Result<(), eframe::Error> {
    if !has_display_server() {
        eprintln!(
            "No graphical display server detected (DISPLAY, WAYLAND_DISPLAY, and WAYLAND_SOCKET are unset).\n\
             The SEG-Y viewer requires an X11 or Wayland session to show the UI.\n\
             Skipping UI launch."
        );
        return Ok(());
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 900.0])
            .with_title("SEG-Y 2D Viewer with First Break Picking (Rust)"),
        multisampling: 4,
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };

    eframe::run_native(
        "SEG-Y 2D Viewer",
        options,
        Box::new(|cc| Box::new(SegyViewerApp::new(cc))),
    )
}

fn has_display_server() -> bool {
    env::var_os("DISPLAY").is_some()
        || env::var_os("WAYLAND_DISPLAY").is_some()
        || env::var_os("WAYLAND_SOCKET").is_some()
}

struct SegyViewerApp {
    segy_reader: SegyReader,
    picking_manager: PickingManager,
    gl_renderer: Option<Arc<Mutex<GlRenderer>>>,

    // UI state
    filename: String,
    colormap: String,
    show_picks: bool,
    picking_enabled: bool,
    algorithm: Algorithm,

    // View state
    zoom: f32,
    offset_x: f32,
    offset_y: f32,
    last_mouse_pos: Option<egui::Pos2>,
    is_panning: bool,

    // Info
    status_message: String,
    mouse_trace: i32,
    mouse_sample: f32,
}

impl SegyViewerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Initialize renderer (GL context은 보관하지 않음)
        let gl_renderer = cc.gl.as_ref().map(|gl| unsafe {
            let mut renderer = GlRenderer::new(gl);
            // 초기 텍스처나 셰이더 세팅 가능
            Arc::new(Mutex::new(renderer))
        });

        Self {
            segy_reader: SegyReader::new(),
            picking_manager: PickingManager::new(),
            gl_renderer,
            filename: String::from("No file loaded"),
            colormap: String::from("seismic"),
            show_picks: true,
            picking_enabled: true,
            algorithm: Algorithm::StaLta,
            zoom: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
            last_mouse_pos: None,
            is_panning: false,
            status_message: String::from("Ready"),
            mouse_trace: -1,
            mouse_sample: -1.0,
        }
    }

    fn open_file(&mut self, path: String) {
        self.status_message = format!("Loading {}...", path);

        match self.segy_reader.load_file(&path) {
            Ok(_) => {
                self.filename = path.clone();
                self.picking_manager.set_num_traces(self.segy_reader.num_traces);
                self.picking_manager.clear_picks();

                self.status_message = format!(
                    "Loaded: {} traces, {} samples",
                    self.segy_reader.num_traces,
                    self.segy_reader.num_samples
                );

                // GPU 업로드 대신 데이터 저장 표시만
                // 실제 GL 업로드는 렌더링 시점에서 수행됨
            }
            Err(e) => {
                self.status_message = format!("Error: {}", e);
            }
        }
    }

    fn auto_pick(&mut self) {
        if self.segy_reader.data.is_empty() {
            self.status_message = String::from("No data loaded");
            return;
        }

        self.status_message = format!("Running auto picking with {:?}...", self.algorithm);

        let picks = AutoPicker::pick_all_traces(
            &self.segy_reader.raw_data,
            self.segy_reader.num_traces,
            self.segy_reader.num_samples,
            self.algorithm,
        );

        self.picking_manager.set_picks(picks.clone());

        self.status_message = format!("Auto picking completed: {} picks", picks.len());
    }

    fn screen_to_data_coords(
        &self,
        screen_pos: egui::Pos2,
        rect: egui::Rect,
    ) -> Option<(usize, f32)> {
        let width = rect.width();
        let height = rect.height();

        if width <= 0.0 || height <= 0.0 {
            return None;
        }

        let norm_x = ((screen_pos.x - rect.left()) / width) * 2.0 - 1.0;
        let norm_y = 1.0 - ((screen_pos.y - rect.top()) / height) * 2.0;
        let norm_x = (norm_x - self.offset_x) / self.zoom;
        let norm_y = (norm_y - self.offset_y) / self.zoom;

        let trace_idx = ((norm_x + 1.0) / 2.0 * self.segy_reader.num_traces as f32) as i32;
        let sample_idx = (norm_y + 1.0) / 2.0 * self.segy_reader.num_samples as f32;

        if trace_idx >= 0
            && (trace_idx as usize) < self.segy_reader.num_traces
            && sample_idx >= 0.0
            && sample_idx < self.segy_reader.num_samples as f32
        {
            Some((trace_idx as usize, sample_idx))
        } else {
            None
        }
    }

    fn get_transform_matrix(&self) -> [f32; 16] {
        [
            self.zoom, 0.0, 0.0, 0.0,
            0.0, self.zoom, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            self.offset_x, self.offset_y, 0.0, 1.0,
        ]
    }
}

impl eframe::App for SegyViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) = ui.allocate_exact_size(
                ui.available_size(),
