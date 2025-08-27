use minifb::{Key, Window, WindowOptions};
use treedee::{Point, TreeBuilder, Vec3};

const WIDTH: usize = 640;
const HEIGHT: usize = 320;

fn main() {
    let mut window = Window::new(
        "Edge test - ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions {
            scale_mode: minifb::ScaleMode::AspectRatioStretch,
            ..WindowOptions::default()
        },
    )
    .unwrap();

    // Limit to max ~60 fps update rate
    window.set_target_fps(60);

    let mut renderer = treedee::BufRenderer::<WIDTH, HEIGHT>::default();

    let mut tree_builder = TreeBuilder::default();

    for _ in 0..=256 {
        tree_builder.add_point(rand::random_range(
            Vec3::ZERO..Vec3 {
                x: 10.0,
                y: 10.0,
                z: 5.0,
            },
        ));
    }

    let scene = treedee::Scene::<treedee::SceneTypeEdge>::new_from_list(
        tree_builder
            .finish()
            .iter()
            .map(|v| Point {
                color: rand::random_range(
                    Vec3::ZERO..Vec3 {
                        x: 1.0,
                        y: 1.0,
                        z: 1.0,
                    },
                ),
                pos: *v,
            })
            .collect::<Vec<Point>>(),
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
