// OpenGL Renderer Module
use glow::HasContext;
use std::sync::Arc;

pub struct GlRenderer {
    gl: Arc<glow::Context>,
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    texture: glow::Texture,
    texture_width: i32,
    texture_height: i32,
}

impl GlRenderer {
    pub unsafe fn new(gl: Arc<glow::Context>) -> Self {
        // Create shader program
        let vertex_shader_source = r#"
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            uniform mat4 transform;
            void main() {
                gl_Position = transform * vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
        "#;

        let fragment_shader_source = r#"
            #version 330 core
            in vec2 TexCoord;
            out vec4 FragColor;
            uniform sampler2D texture1;
            void main() {
                FragColor = texture(texture1, TexCoord);
            }
        "#;

        let program = Self::create_program(&gl, vertex_shader_source, fragment_shader_source);

        // Create quad vertices
        #[rustfmt::skip]
        let vertices: [f32; 16] = [
            // positions   // texture coords
            -1.0, -1.0,    0.0, 1.0,
             1.0, -1.0,    1.0, 1.0,
             1.0,  1.0,    1.0, 0.0,
            -1.0,  1.0,    0.0, 0.0,
        ];

        let vao = gl.create_vertex_array().unwrap();
        let vbo = gl.create_buffer().unwrap();

        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytemuck::cast_slice(&vertices),
            glow::STATIC_DRAW,
        );

        // Position attribute
        gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 16, 0);
        gl.enable_vertex_attrib_array(0);

        // Texture coord attribute
        gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 16, 8);
        gl.enable_vertex_attrib_array(1);

        let texture = gl.create_texture().unwrap();

        Self {
            gl,
            program,
            vao,
            vbo,
            texture,
            texture_width: 0,
            texture_height: 0,
        }
    }

    unsafe fn create_program(
        gl: &glow::Context,
        vertex_src: &str,
        fragment_src: &str,
    ) -> glow::Program {
        let program = gl.create_program().unwrap();

        let vertex_shader = gl.create_shader(glow::VERTEX_SHADER).unwrap();
        gl.shader_source(vertex_shader, vertex_src);
        gl.compile_shader(vertex_shader);
        if !gl.get_shader_compile_status(vertex_shader) {
            panic!("{}", gl.get_shader_info_log(vertex_shader));
        }

        let fragment_shader = gl.create_shader(glow::FRAGMENT_SHADER).unwrap();
        gl.shader_source(fragment_shader, fragment_src);
        gl.compile_shader(fragment_shader);
        if !gl.get_shader_compile_status(fragment_shader) {
            panic!("{}", gl.get_shader_info_log(fragment_shader));
        }

        gl.attach_shader(program, vertex_shader);
        gl.attach_shader(program, fragment_shader);
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!("{}", gl.get_program_info_log(program));
        }

        gl.delete_shader(vertex_shader);
        gl.delete_shader(fragment_shader);

        program
    }

    pub unsafe fn upload_texture(&mut self, data: &[Vec<f32>], colormap: &str) {
        if data.is_empty() || data[0].is_empty() {
            return;
        }

        let height = data.len();
        let width = data[0].len();

        // Convert to RGB texture
        let mut texture_data = vec![0u8; width * height * 3];

        for y in 0..height {
            for x in 0..width {
                let value = data[y][x].clamp(-1.0, 1.0);
                let (r, g, b) = Self::apply_colormap(value, colormap);

                let idx = (y * width + x) * 3;
                texture_data[idx] = r;
                texture_data[idx + 1] = g;
                texture_data[idx + 2] = b;
            }
        }

        self.texture_width = width as i32;
        self.texture_height = height as i32;

        self.gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
        self.gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR as i32,
        );
        self.gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );

        self.gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGB as i32,
            width as i32,
            height as i32,
            0,
            glow::RGB,
            glow::UNSIGNED_BYTE,
            Some(&texture_data),
        );
    }

    fn apply_colormap(value: f32, colormap: &str) -> (u8, u8, u8) {
        if !value.is_finite() {
            return (0, 0, 0);
        }

        let value = value.clamp(-1.0, 1.0);

        match colormap {
            "seismic" => {
                if value < 0.0 {
                    let t = value + 1.0;
                    let r = (t * 255.0) as u8;
                    let g = (t * 255.0) as u8;
                    let b = 255;
                    (r, g, b)
                } else {
                    let t = 1.0 - value;
                    let r = 255;
                    let g = (t * 255.0) as u8;
                    let b = (t * 255.0) as u8;
                    (r, g, b)
                }
            }
            "grayscale" => {
                let v = ((value + 1.0) / 2.0 * 255.0) as u8;
                (v, v, v)
            }
            _ => {
                let v = ((value + 1.0) / 2.0 * 255.0) as u8;
                (v, v, v)
            }
        }
    }

    pub unsafe fn render(&self, transform: &[f32; 16]) {
        self.gl.use_program(Some(self.program));

        // Set transform uniform
        let transform_loc = self
            .gl
            .get_uniform_location(self.program, "transform")
            .unwrap();
        self.gl
            .uniform_matrix_4_f32_slice(Some(&transform_loc), false, transform);

        // Bind texture
        self.gl.active_texture(glow::TEXTURE0);
        self.gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));

        // Draw quad
        self.gl.bind_vertex_array(Some(self.vao));
        self.gl.draw_arrays(glow::TRIANGLE_FAN, 0, 4);
    }

    pub unsafe fn cleanup(&self) {
        self.gl.delete_program(self.program);
        self.gl.delete_vertex_array(self.vao);
        self.gl.delete_buffer(self.vbo);
        self.gl.delete_texture(self.texture);
    }
}
