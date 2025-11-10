// Main Application with eframe + egui
mod auto_picking;
mod picking_manager;
mod segy_reader;

use auto_picking::{Algorithm, AutoPicker};
use eframe::egui;
use picking_manager::PickingManager;
use segy_reader::SegyReader;

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
            colormap: String::from("grayscale"),
            show_picks: true,
            picking_enabled: true,
            algorithm: Algorithm::StaLta,
            zoom: 1.0,
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

    fn bilinear_interpolate(&self, x: f32, y: f32) -> f32 {
        let orig_height = self.segy_reader.num_samples;
        let orig_width = self.segy_reader.num_traces;

        // Get integer and fractional parts
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(orig_width - 1);
        let y1 = (y0 + 1).min(orig_height - 1);

        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        // Get the four surrounding values
        let v00 = self.segy_reader.data[y0][x0];
        let v10 = self.segy_reader.data[y0][x1];
        let v01 = self.segy_reader.data[y1][x0];
        let v11 = self.segy_reader.data[y1][x1];

        // Handle NaN values - if any value is NaN, fall back to nearest neighbor
        if !v00.is_finite() || !v10.is_finite() || !v01.is_finite() || !v11.is_finite() {
            return v00;
        }

        // Bilinear interpolation formula
        let v0 = v00 * (1.0 - dx) + v10 * dx;
        let v1 = v01 * (1.0 - dx) + v11 * dx;
        v0 * (1.0 - dy) + v1 * dy
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        if self.segy_reader.data.is_empty() {
            return;
        }

        let orig_height = self.segy_reader.num_samples;
        let orig_width = self.segy_reader.num_traces;

        // GPU texture size limit - increased for better quality
        const MAX_TEXTURE_SIZE: usize = 16384;
        const MIN_TEXTURE_SIZE: usize = 2048;

        // Calculate target texture size for high quality
        let width = if orig_width > MAX_TEXTURE_SIZE {
            MAX_TEXTURE_SIZE
        } else if orig_width < MIN_TEXTURE_SIZE {
            MIN_TEXTURE_SIZE.min(orig_width)
        } else {
            orig_width
        };

        let height = if orig_height > MAX_TEXTURE_SIZE {
            MAX_TEXTURE_SIZE
        } else if orig_height < MIN_TEXTURE_SIZE {
            MIN_TEXTURE_SIZE.min(orig_height)
        } else {
            orig_height
        };

        let width_scale = orig_width as f32 / width as f32;
        let height_scale = orig_height as f32 / height as f32;

        println!("Converting {}x{} data to {}x{} texture (scale: {:.2}x{:.2})...",
                 orig_width, orig_height, width, height, width_scale, height_scale);
        println!("Using bilinear interpolation for high-quality display");

        let mut pixels = vec![egui::Color32::BLACK; width * height];

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut nan_count = 0;

        for y in 0..height {
            let orig_y_f = (y as f32 * height_scale).min(orig_height as f32 - 1.0);
            for x in 0..width {
                let orig_x_f = (x as f32 * width_scale).min(orig_width as f32 - 1.0);

                // Use bilinear interpolation
                let value = self.bilinear_interpolate(orig_x_f, orig_y_f);

                if value.is_finite() {
                    min_val = min_val.min(value);
                    max_val = max_val.max(value);
                } else {
                    nan_count += 1;
                }

                let clamped_value = value.clamp(-1.0, 1.0);
                let color = self.apply_colormap(clamped_value);
                pixels[y * width + x] = color;
            }
        }

        println!("Data range: [{:.3}, {:.3}], NaN count: {}/{}", min_val, max_val, nan_count, width * height);

        let color_image = egui::ColorImage {
            size: [width, height],
            pixels,
        };

        println!("Loading {}x{} texture to GPU...", width, height);
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

    fn screen_to_data_coords_simple(&self, screen_pos: egui::Pos2, rect: egui::Rect, display_size: egui::Vec2) -> Option<(usize, f32)> {
        if display_size.x <= 0.0 || display_size.y <= 0.0 {
            return None;
        }

        // Convert screen position to normalized coordinates [0, 1]
        let norm_x = (screen_pos.x - rect.left()) / display_size.x;
        let norm_y = (screen_pos.y - rect.top()) / display_size.y;

        // Convert to trace and sample indices
        let trace_idx = (norm_x * self.segy_reader.num_traces as f32) as i32;
        let sample_idx = norm_y * self.segy_reader.num_samples as f32;

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

    fn draw_picks(&self, ui: &mut egui::Ui, rect: egui::Rect, display_size: egui::Vec2) {
        let picks = self.picking_manager.get_picks();

        if picks.is_empty() {
            return;
        }

        let painter = ui.painter();

        // Draw individual pick points
        for (trace_idx, sample_idx) in picks.iter() {
            let x = rect.left() + (*trace_idx as f32 / self.segy_reader.num_traces as f32) * display_size.x;
            let y = rect.top() + (sample_idx / self.segy_reader.num_samples as f32) * display_size.y;

            let pos = egui::pos2(x, y);

            // Draw a circle at pick point
            painter.circle_filled(pos, 3.0, egui::Color32::RED);
            painter.circle_stroke(pos, 3.0, egui::Stroke::new(1.0, egui::Color32::WHITE));
        }

        // Draw interpolated line
        let interpolated = self.picking_manager.get_interpolated();
        let mut points = Vec::new();

        for trace_idx in 0..self.segy_reader.num_traces {
            let sample_idx = interpolated[trace_idx];
            if sample_idx >= 0.0 {
                let x = rect.left() + (trace_idx as f32 / self.segy_reader.num_traces as f32) * display_size.x;
                let y = rect.top() + (sample_idx / self.segy_reader.num_samples as f32) * display_size.y;
                points.push(egui::pos2(x, y));
            }
        }

        // Draw the line connecting picks
        if points.len() > 1 {
            painter.add(egui::Shape::line(points, egui::Stroke::new(2.0, egui::Color32::YELLOW)));
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

                if ui.button("Reset Zoom").clicked() {
                    self.zoom = 1.0;
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
            if let Some(texture) = &self.texture {
                println!("[Render] Texture exists: {}x{}", texture.size()[0], texture.size()[1]);

                let available_size = ui.available_size();
                let texture_id = texture.id();
                let img_size = texture.size_vec2();

                // Scale to fill screen height (width will be larger and scrollable)
                let height_scale = available_size.y / img_size.y;

                // Apply zoom on top of height scale
                let display_size = img_size * height_scale * self.zoom;

                println!("[Render] Display size: {:?}, height_scale: {:.3}, zoom: {:.2}",
                         display_size, height_scale, self.zoom);

                // Use ScrollArea for horizontal/vertical scrolling
                egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let (rect, response) = ui.allocate_exact_size(
                            display_size,
                            egui::Sense::click_and_drag(),
                        );

                        // Handle mouse events for picking
                        if response.clicked() && self.picking_enabled {
                            if let Some(pos) = response.interact_pointer_pos() {
                                if let Some((trace_idx, sample_idx)) = self.screen_to_data_coords_simple(pos, rect, display_size) {
                                    self.picking_manager.add_pick(trace_idx, sample_idx);
                                    println!("Pick added at trace: {}, sample: {:.1}", trace_idx, sample_idx);
                                }
                            }
                        }

                        // Mouse wheel for zoom (Ctrl + wheel)
                        let zoom_delta = ui.input(|i| i.zoom_delta());
                        if zoom_delta != 1.0 {
                            self.zoom *= zoom_delta;
                            self.zoom = self.zoom.clamp(0.5, 10.0);
                        }

                        // Track mouse position
                        if let Some(pos) = response.hover_pos() {
                            if let Some((trace_idx, sample_idx)) = self.screen_to_data_coords_simple(pos, rect, display_size) {
                                self.mouse_trace = trace_idx as i32;
                                self.mouse_sample = sample_idx;
                            }
                        }

                        // Draw the image
                        ui.painter().image(
                            texture_id,
                            rect,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );

                        // Draw picks
                        if self.show_picks {
                            self.draw_picks(ui, rect, display_size);
                        }
                    });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("No data loaded. Click 'Open SEG-Y' to load a file.");
                });
            }
        });
    }
}
