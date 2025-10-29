// Main Application with eframe + egui + OpenGL
mod auto_picking;
mod gl_renderer;
mod picking_manager;
mod segy_reader;

use auto_picking::{Algorithm, AutoPicker};
use eframe::egui;
use gl_renderer::GlRenderer;
use glow::HasContext;
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
        // Initialize OpenGL renderer
        let gl_renderer = cc.gl.as_ref().map(|gl| {
            Arc::new(Mutex::new(unsafe {
                GlRenderer::new(gl.clone())
            }))
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

                // Upload texture to GPU
                if let Some(renderer) = &self.gl_renderer {
                    if let Ok(mut r) = renderer.lock() {
                        unsafe {
                            r.upload_texture(&self.segy_reader.data, &self.colormap);
                        }
                    }
                }
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

    fn screen_to_data_coords(&self, screen_pos: egui::Pos2, rect: egui::Rect) -> Option<(usize, f32)> {
        let width = rect.width();
        let height = rect.height();

        if width <= 0.0 || height <= 0.0 {
            return None;
        }

        // Screen to normalized [-1, 1]
        let norm_x = ((screen_pos.x - rect.left()) / width) * 2.0 - 1.0;
        let norm_y = 1.0 - ((screen_pos.y - rect.top()) / height) * 2.0;

        // Apply inverse transform
        let norm_x = (norm_x - self.offset_x) / self.zoom;
        let norm_y = (norm_y - self.offset_y) / self.zoom;

        // To data coordinates
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
        #[rustfmt::skip]
        let transform = [
            self.zoom, 0.0, 0.0, 0.0,
            0.0, self.zoom, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            self.offset_x, self.offset_y, 0.0, 1.0,
        ];
        transform
    }
}

impl eframe::App for SegyViewerApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Top panel - toolbar
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📁 Open SEG-Y").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("SEG-Y", &["sgy", "segy"])
                        .pick_file()
                    {
                        self.open_file(path.display().to_string());
                    }
                }

                ui.separator();

                if ui.button("Reset View").clicked() {
                    self.zoom = 1.0;
                    self.offset_x = 0.0;
                    self.offset_y = 0.0;
                }

                ui.separator();

                if ui.button("💾 Save Picks").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("CSV", &["csv"])
                        .save_file()
                    {
                        if let Err(e) = self.picking_manager.save_to_file(&path.display().to_string()) {
                            self.status_message = format!("Error saving: {}", e);
                        } else {
                            self.status_message = String::from("Picks saved");
                        }
                    }
                }

                if ui.button("📂 Load Picks").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("CSV", &["csv"])
                        .pick_file()
                    {
                        if let Err(e) = self.picking_manager.load_from_file(&path.display().to_string()) {
                            self.status_message = format!("Error loading: {}", e);
                        } else {
                            self.status_message = String::from("Picks loaded");
                        }
                    }
                }

                if ui.button("🗑 Clear Picks").clicked() {
                    self.picking_manager.clear_picks();
                    self.status_message = String::from("Picks cleared");
                }

                ui.separator();

                if ui.button("🤖 Auto Pick").clicked() {
                    self.auto_pick();
                }
            });
        });

        // Bottom panel - status bar
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status_message);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("Trace: {}, Sample: {:.1}", self.mouse_trace, self.mouse_sample));
                });
            });
        });

        // Right panel - control panel
        egui::SidePanel::right("control_panel")
            .default_width(300.0)
            .show(ctx, |ui| {
                ui.heading("Control Panel");
                ui.separator();

                ui.label("📄 File Information");
                ui.label(&self.filename);
                ui.add_space(10.0);

                ui.separator();
                ui.label("🎯 Picking Options");
                ui.checkbox(&mut self.picking_enabled, "Enable Manual Picking");
                ui.checkbox(&mut self.show_picks, "Show Picks");
                self.picking_manager.set_enabled(self.picking_enabled);
                ui.add_space(10.0);

                ui.separator();
                ui.label("🎨 Display Options");
                egui::ComboBox::from_label("Colormap")
                    .selected_text(&self.colormap)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.colormap, "seismic".to_string(), "Seismic");
                        ui.selectable_value(&mut self.colormap, "grayscale".to_string(), "Grayscale");
                    });
                ui.add_space(10.0);

                ui.separator();
                ui.label("🤖 Auto Picking");
                egui::ComboBox::from_label("Algorithm")
                    .selected_text(format!("{:?}", self.algorithm))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.algorithm, Algorithm::StaLta, "STA/LTA");
                        ui.selectable_value(&mut self.algorithm, Algorithm::EnergyRatio, "Energy Ratio");
                        ui.selectable_value(&mut self.algorithm, Algorithm::Aic, "AIC");
                    });

                if ui.button("▶ Run Auto Pick").clicked() {
                    self.auto_pick();
                }

                ui.add_space(20.0);
                ui.separator();
                ui.label("📖 Controls:");
                ui.label("• Left Click: Pick");
                ui.label("• Mouse Wheel: Zoom");
                ui.label("• Right Drag: Pan");
            });

        // Central panel - OpenGL viewer
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) = ui.allocate_exact_size(
                ui.available_size(),
                egui::Sense::click_and_drag(),
            );

            // Handle mouse events
            if response.clicked() && self.picking_enabled {
                if let Some(pos) = response.interact_pointer_pos() {
                    if let Some((trace_idx, sample_idx)) = self.screen_to_data_coords(pos, rect) {
                        self.picking_manager.add_pick(trace_idx, sample_idx);
                    }
                }
            }

            if response.dragged_by(egui::PointerButton::Secondary) {
                let delta = response.drag_delta();
                self.offset_x += delta.x / rect.width() * 2.0 / self.zoom;
                self.offset_y -= delta.y / rect.height() * 2.0 / self.zoom;
            }

            // Scroll for zoom
            let scroll_delta = ui.input(|i| i.smooth_scroll_delta);
            if scroll_delta.y != 0.0 {
                let zoom_factor = if scroll_delta.y > 0.0 { 1.1 } else { 0.9 };
                self.zoom *= zoom_factor;
                self.zoom = self.zoom.clamp(0.1, 10.0);
            }

            // Update mouse position
            if let Some(pos) = response.hover_pos() {
                if let Some((trace_idx, sample_idx)) = self.screen_to_data_coords(pos, rect) {
                    self.mouse_trace = trace_idx as i32;
                    self.mouse_sample = sample_idx;
                }
            }

            // OpenGL rendering
            let has_data = !self.segy_reader.data.is_empty();
            let transform = self.get_transform_matrix();
            let picks = self.picking_manager.get_interpolated().to_vec();
            let show_picks = self.show_picks;
            let num_traces = self.segy_reader.num_traces;
            let gl_renderer = self.gl_renderer.clone();

            let callback = egui::PaintCallback {
                rect,
                callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                    let gl = painter.gl();
                    unsafe {
                        // Clear
                        gl.clear_color(0.0, 0.0, 0.0, 1.0);
                        gl.clear(glow::COLOR_BUFFER_BIT);

                        // Render SEG-Y data if available
                        if has_data {
                            if let Some(renderer) = &gl_renderer {
                                if let Ok(r) = renderer.lock() {
                                    r.render(&transform);

                                    // TODO: Implement picking line rendering with modern OpenGL
                                    // (requires separate shader and VBO)
                                }
                            }
                        }
                    }
                })),
            };

            ui.painter().add(callback);
        });

        ctx.request_repaint();
    }
}
