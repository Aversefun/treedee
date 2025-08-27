use minifb::{Key, Window, WindowOptions};
use treedee::{Point, Vec3};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;

fn main() {
    let mut window = Window::new(
        "Point test - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.set_target_fps(60);

    let mut renderer = treedee::BufRenderer::<WIDTH, HEIGHT>::default();

    let scene = treedee::Scene::<treedee::SceneTypePoint>::new_from_list(
        [Point {
            pos: Vec3::ZERO,
            color: Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        }; 256]
            .map(|_| Point {
                color: rand::random_range(
                    Vec3::ZERO..Vec3 {
                        x: 1.0,
                        y: 1.0,
                        z: 1.0,
                    },
                ),
                pos: rand::random_range(
                    Vec3::ZERO..Vec3 {
                        x: 10.0,
                        y: 10.0,
                        z: 5.0,
                    },
                ),
            }),
    )
    .unwrap();

    scene.render(&mut renderer, Vec3::ZERO, Vec3::ZERO);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        window
            .update_with_buffer(
                renderer
                    .buf
                    .iter()
                    .flat_map(|v| {
                        v.iter()
                            .map(|v| {
                                ((v.x * 255.0) as u32) << 16
                                    | ((v.y * 255.0) as u32) << 8
                                    | (v.z * 255.0) as u32
                            })
                            .collect::<Vec<u32>>()
                    })
                    .collect::<Vec<_>>()
                    .as_slice(),
                WIDTH,
                HEIGHT,
            )
            .unwrap();
    }
}
