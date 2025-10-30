// Main Application with eframe + egui
mod auto_picking;
mod picking_manager;
mod segy_reader;

use auto_picking::{Algorithm, AutoPicker};
use eframe::egui;
use picking_manager::PickingManager;
use segy_reader::SegyReader;
use std::env;

fn main() -> Result<(), eframe::Error> {
    println!("Starting SEG-Y 2D Viewer...");

    // Only check for display server on Linux/Unix
    #[cfg(not(target_os = "windows"))]
    {
        if !has_display_server() {
            eprintln!(
                "No graphical display server detected.\n\
                 The SEG-Y viewer requires a graphical display.\n\
                 Skipping UI launch."
            );
            return Ok(());
        }
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 900.0])
            .with_title("SEG-Y 2D Viewer with First Break Picking (Rust)"),
        ..Default::default()
    };

    println!("Running application...");
    eframe::run_native(
        "SEG-Y 2D Viewer",
        options,
        Box::new(|cc| {
            println!("Creating app...");
            Box::new(SegyViewerApp::new(cc))
        }),
    )
}

#[cfg(not(target_os = "windows"))]
fn has_display_server() -> bool {
    env::var_os("DISPLAY").is_some()
        || env::var_os("WAYLAND_DISPLAY").is_some()
        || env::var_os("WAYLAND_SOCKET").is_some()
}

struct SegyViewerApp {
    segy_reader: SegyReader,
    picking_manager: PickingManager,
    texture: Option<egui::TextureHandle>,
    filename: String,
    colormap: String,
    show_picks: bool,
    picking_enabled: bool,
    algorithm: Algorithm,
    zoom: f32,
    offset_x: f32,
    offset_y: f32,
    status_message: String,
    mouse_trace: i32,
    mouse_sample: f32,
}

impl SegyViewerApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        println!("App created");
        Self {
            segy_reader: SegyReader::new(),
            picking_manager: PickingManager::new(),
            texture: None,
            filename: String::from("No file loaded"),
            colormap: String::from("seismic"),
            show_picks: true,
            picking_enabled: true,
            algorithm: Algorithm::StaLta,
            zoom: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
            status_message: String::from("Ready"),
            mouse_trace: -1,
            mouse_sample: -1.0,
        }
    }

    fn open_file(&mut self, ctx: &egui::Context, path: String) {
        self.status_message = format!("Loading {}...", path);

        match self.segy_reader.load_file(&path) {
            Ok(_) => {
                self.filename = path.clone();
                self.picking_manager.set_num_traces(self.segy_reader.num_traces);
                self.picking_manager.clear_picks();

                self.status_message = format!(
                    "Loaded: {} traces, {} samples",
                    self.segy_reader.num_traces, self.segy_reader.num_samples
                );

                println!("Creating texture from data...");
                self.update_texture(ctx);
            }
            Err(e) => {
                self.status_message = format!("Error: {}", e);
            }
        }
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        if self.segy_reader.data.is_empty() {
            return;
        }

        let height = self.segy_reader.num_samples;
        let width = self.segy_reader.num_traces;

        println!("Converting {}x{} data to image...", width, height);

        let mut pixels = vec![egui::Color32::BLACK; width * height];

        for y in 0..height {
            for x in 0..width {
                let value = self.segy_reader.data[y][x].clamp(-1.0, 1.0);
                let color = self.apply_colormap(value);
                pixels[y * width + x] = color;
            }
        }

        let color_image = egui::ColorImage {
            size: [width, height],
            pixels,
        };

        println!("Loading texture to GPU...");
        self.texture = Some(ctx.load_texture(
            "segy_data",
            color_image,
            egui::TextureOptions::LINEAR,
        ));

        println!("Texture created successfully!");
    }

    fn apply_colormap(&self, value: f32) -> egui::Color32 {
        if !value.is_finite() {
            return egui::Color32::BLACK;
        }

        let value = value.clamp(-1.0, 1.0);

        match self.colormap.as_str() {
            "seismic" => {
                if value < 0.0 {
                    let t = value + 1.0;
                    let r = (t * 255.0) as u8;
                    let g = (t * 255.0) as u8;
                    let b = 255;
                    egui::Color32::from_rgb(r, g, b)
                } else {
                    let t = 1.0 - value;
                    let r = 255;
                    let g = (t * 255.0) as u8;
                    let b = (t * 255.0) as u8;
                    egui::Color32::from_rgb(r, g, b)
                }
            }
            "grayscale" => {
                let v = ((value + 1.0) / 2.0 * 255.0) as u8;
                egui::Color32::from_gray(v)
            }
            _ => {
                let v = ((value + 1.0) / 2.0 * 255.0) as u8;
                egui::Color32::from_gray(v)
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
}

impl eframe::App for SegyViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📁 Open SEG-Y").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("SEG-Y", &["sgy", "segy"])
                        .pick_file()
                    {
                        self.open_file(ctx, path.display().to_string());
                    }
                }

                ui.separator();

                if ui.button("Reset View").clicked() {
                    self.zoom = 1.0;
                    self.offset_x = 0.0;
                    self.offset_y = 0.0;
                }

                ui.separator();

                if ui.button("🤖 Auto Pick").clicked() {
                    self.auto_pick();
                }

                ui.separator();
                ui.label(&self.status_message);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("Trace: {}, Sample: {:.1}", self.mouse_trace, self.mouse_sample));
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let available_size = ui.available_size();

            if let Some(texture) = &self.texture {
                let (rect, response) = ui.allocate_exact_size(
                    available_size,
                    egui::Sense::click_and_drag(),
                );

            if response.dragged_by(egui::PointerButton::Secondary) {
                let delta = response.drag_delta();
                self.offset_x += delta.x / rect.width() * 2.0 / self.zoom;
                self.offset_y -= delta.y / rect.height() * 2.0 / self.zoom;
            }

            let scroll_delta = ui.input(|i| i.smooth_scroll_delta);
            if scroll_delta.y != 0.0 {
                let zoom_factor = if scroll_delta.y > 0.0 { 1.1 } else { 0.9 };
                self.zoom *= zoom_factor;
                self.zoom = self.zoom.clamp(0.1, 10.0);
            }

            if let Some(pos) = response.hover_pos() {
                if let Some((trace_idx, sample_idx)) = self.screen_to_data_coords(pos, rect) {
                    self.mouse_trace = trace_idx as i32;
                    self.mouse_sample = sample_idx;
                }
            }

            let has_data = !self.segy_reader.data.is_empty();
            let transform = self.get_transform_matrix();
            let gl_renderer = self.gl_renderer.clone();
            let data_clone = if has_data {
                Some((self.segy_reader.data.clone(), self.colormap.clone()))
            } else {
                None
            };

            let callback = egui::PaintCallback {
                rect,
                callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                    let gl = painter.gl();
                    unsafe {
                        gl.clear_color(0.0, 0.0, 0.0, 1.0);
                        gl.clear(glow::COLOR_BUFFER_BIT);

                        if let Some(renderer) = &gl_renderer {
                            if let Some((data, colormap)) = &data_clone {
                                if let Some(mut r) = renderer.try_lock() {
                                    r.upload_texture(gl, data, &colormap);
                                    // Upload texture only if not already uploaded
                                    if r.texture_width == 0 {
                                        r.upload_texture(gl, data, colormap);
                                    }
                                }
                            }

                            if has_data {
                                if let Some(r) = renderer.try_lock() {
                                    r.render(gl, &transform);
                                }
                            }
                        }
                    }
                })),
            };  
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

                let scroll_delta = ui.input(|i| i.smooth_scroll_delta);
                if scroll_delta.y != 0.0 {
                    let zoom_factor = if scroll_delta.y > 0.0 { 1.1 } else { 0.9 };
                    self.zoom *= zoom_factor;
                    self.zoom = self.zoom.clamp(0.1, 10.0);
                }

                if let Some(pos) = response.hover_pos() {
                    if let Some((trace_idx, sample_idx)) = self.screen_to_data_coords(pos, rect) {
                        self.mouse_trace = trace_idx as i32;
                        self.mouse_sample = sample_idx;
                    }
                }

                // Calculate scaled size
                let img_size = texture.size_vec2();
                let scale = (rect.width() / img_size.x).min(rect.height() / img_size.y) * self.zoom;
                let scaled_size = img_size * scale;

                // Center the image
                let offset = egui::vec2(
                    rect.center().x - scaled_size.x / 2.0 + self.offset_x * scaled_size.x / 2.0,
                    rect.center().y - scaled_size.y / 2.0 + self.offset_y * scaled_size.y / 2.0,
                );

                let image_rect = egui::Rect::from_min_size(
                    egui::pos2(offset.x, offset.y),
                    scaled_size,
                );

                // Draw the image
                ui.painter().image(
                    texture.id(),
                    image_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("No data loaded. Click 'Open SEG-Y' to load a file.");
                });
            }
        });
    }
}
