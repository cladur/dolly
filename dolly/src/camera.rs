use winit::event::*;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

pub struct Camera {
    pub position: cgmath::Point3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    fn get_front(&self) -> cgmath::Vector3<f32> {
        cgmath::Vector3 {
            x: self.yaw.to_radians().cos() * self.pitch.to_radians().cos(),
            y: self.pitch.to_radians().sin(),
            z: self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
        }
    }

    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let front = self.get_front();
        let up = cgmath::Vector3::unit_y();
        let view = cgmath::Matrix4::look_at_rh(self.position, self.position + front, up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

pub struct CameraController {
    speed: f32,
    sensitivity: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_down_pressed: bool,
    is_up_pressed: bool,
    pub controls_camera: bool,
    pub mouse_delta: (f64, f64),
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            sensitivity: 0.12,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_down_pressed: false,
            is_up_pressed: false,
            controls_camera: false,
            mouse_delta: (0.0, 0.0),
        }
    }

    pub fn process_events(&mut self, window: &winit::window::Window, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseInput {
                device_id,
                state,
                button,
                ..
            } => {
                if state == &ElementState::Pressed && button == &MouseButton::Right {
                    self.controls_camera = true;
                    window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
                    window.set_cursor_visible(false);
                } else {
                    self.controls_camera = false;
                    window.set_cursor_grab(winit::window::CursorGrabMode::None);
                    window.set_cursor_visible(true);
                }
                true
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::E => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::Q => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        if !self.controls_camera {
            return;
        }

        use cgmath::InnerSpace;
        // Compute forward from yaw and pitch
        let forward = camera.get_front();
        let right = forward.cross(cgmath::Vector3::unit_y()).normalize();

        if self.is_forward_pressed {
            camera.position += forward * self.speed;
        }

        if self.is_backward_pressed {
            camera.position -= forward * self.speed;
        }

        if self.is_left_pressed {
            camera.position -= right * self.speed;
        }

        if self.is_right_pressed {
            camera.position += right * self.speed;
        }

        if self.is_down_pressed {
            camera.position -= cgmath::Vector3::unit_y() * self.speed;
        }

        if self.is_up_pressed {
            camera.position += cgmath::Vector3::unit_y() * self.speed;
        }

        camera.yaw += self.mouse_delta.0 as f32 * self.sensitivity;
        camera.pitch -= self.mouse_delta.1 as f32 * self.sensitivity;

        camera.pitch = camera.pitch.clamp(-89.0, 89.0);

        self.mouse_delta = (0.0, 0.0);
    }
}
