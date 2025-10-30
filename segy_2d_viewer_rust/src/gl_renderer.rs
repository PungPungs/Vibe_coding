// OpenGL Renderer Module
use glow::HasContext;

pub struct GlRenderer {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    texture: glow::Texture,
    pub texture_width: i32,
    pub texture_height: i32,
}

impl GlRenderer {
    /// GL context를 인자로 받아 초기화
    pub unsafe fn new(gl: &glow::Context) -> Self {
        // === Shader setup ===
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

        let program = Self::create_program(gl, vertex_shader_source, fragment_shader_source);

        // === Quad vertices ===
        #[rustfmt::skip]
        let vertices: [f32; 16] = [
            // positions   // tex coords
            -1.0, -1.0,    0.0, 1.0,
             1.0, -1.0,    1.0, 1.0,
             1.0,  1.0,    1.0, 0.0,
            -1.0,  1.0,    0.0, 0.0,
        ];

        let vao = gl.create_vertex_array().unwrap();
        let vbo = gl.create_buffer().unwrap();

        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&vertices), glow::STATIC_DRAW);

        gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 16, 0);
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 16, 8);
        gl.enable_vertex_attrib_array(1);

        let texture = gl.create_texture().unwrap();

        Self {
            program,
            vao,
            vbo,
            texture,
            texture_width: 0,
            texture_height: 0,
        }
    }

    unsafe fn create_program(gl: &glow::Context, vertex_src: &str, fragment_src: &str) -> glow::Program {
        let program = gl.create_program().unwrap();

        let vs = gl.create_shader(glow::VERTEX_SHADER).unwrap();
        gl.shader_source(vs, vertex_src);
        gl.compile_shader(vs);
        assert!(gl.get_shader_compile_status(vs), "Vertex shader error: {}", gl.get_shader_info_log(vs));

        let fs = gl.create_shader(glow::FRAGMENT_SHADER).unwrap();
        gl.shader_source(fs, fragment_src);
        gl.compile_shader(fs);
        assert!(gl.get_shader_compile_status(fs), "Fragment shader error: {}", gl.get_shader_info_log(fs));

        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.link_program(program);
        assert!(gl.get_program_link_status(program), "Program link error: {}", gl.get_program_info_log(program));

        gl.delete_shader(vs);
        gl.delete_shader(fs);

        program
    }

    /// 텍스처 업로드 (GL context는 매번 인자로 전달)
    pub unsafe fn upload_texture(&mut self, gl: &glow::Context, data: &[Vec<f32>], colormap: &str) {
        println!("[GlRenderer] upload_texture called");

        if data.is_empty() || data[0].is_empty() {
            println!("[GlRenderer] WARNING: Empty data!");
            return;
        }

        let height = data.len();
        let width = data[0].len();
        println!("[GlRenderer] Texture dimensions: {}x{}", width, height);
        let mut texture_data = vec![0u8; width * height * 3];

        for y in 0..height {
            for x in 0..width {
                let value = data[y][x].clamp(-1.0, 1.0);
                let (r, g, b) = Self::apply_colormap(value, colormap);
                let idx = (y * width + x) * 3;
                texture_data[idx..idx + 3].copy_from_slice(&[r, g, b]);
            }
        }

        self.texture_width = width as i32;
        self.texture_height = height as i32;

        gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);

        gl.tex_image_2d(
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
        let value = value.clamp(-1.0, 1.0);
        match colormap {
            "seismic" => {
                if value < 0.0 {
                    let t = value + 1.0;
                    ((t * 255.0) as u8, (t * 255.0) as u8, 255)
                } else {
                    let t = 1.0 - value;
                    (255, (t * 255.0) as u8, (t * 255.0) as u8)
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

    /// 렌더링 함수 - GL context를 외부에서 받음
    pub unsafe fn render(&self, gl: &glow::Context, transform: &[f32; 16]) {
        println!("[GlRenderer] render called, texture size: {}x{}", self.texture_width, self.texture_height);

        gl.use_program(Some(self.program));

        if let Some(loc) = gl.get_uniform_location(self.program, "transform") {
            gl.uniform_matrix_4_f32_slice(Some(&loc), false, transform);
        } else {
            println!("[GlRenderer] WARNING: transform uniform not found!");
        }

        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));

        gl.bind_vertex_array(Some(self.vao));
        gl.draw_arrays(glow::TRIANGLE_FAN, 0, 4);

        println!("[GlRenderer] render complete");
    }

    pub unsafe fn cleanup(&self, gl: &glow::Context) {
        gl.delete_program(self.program);
        gl.delete_vertex_array(self.vao);
        gl.delete_buffer(self.vbo);
        gl.delete_texture(self.texture);
    }
}
