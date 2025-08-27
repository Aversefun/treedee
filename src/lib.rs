//! New Rust Project
#![warn(
    missing_docs,
    clippy::missing_docs_in_private_items,
    clippy::pedantic,
    clippy::all,
    clippy::ignore_without_reason
)]
#![allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use std::fmt::Debug;

use anyhow::ensure;

use rand::distr::uniform::{SampleBorrow, SampleUniform, UniformSampler};
use trig_const::{cos, sin};

/// Something to build a tree.
#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct TreeBuilder {
    /// The points.
    pub points: Vec<Vec3>,
}

impl TreeBuilder {
    /// Add a point to this `TreeBuilder`.
    #[expect(clippy::missing_panics_doc, reason = "false positive")]
    pub fn add_point(&mut self, point: Vec3) -> &mut Self {
        self.points.push(point);
        if !self.points.is_empty() && self.points.len().is_multiple_of(2) {
            self.points.push(*self.points.last().unwrap());
        }
        self
    }
    /// Finish this `TreeBuilder.`
    #[must_use]
    #[expect(clippy::missing_panics_doc, reason = "false positive")]
    pub fn finish(mut self) -> Vec<Vec3> {
        if !self.points.is_empty() {
            self.points.push(*self.points.last().unwrap());
        }
        self.points
    }
}

/// A 3D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Vec2 {
    /// The X component.
    pub x: f64,
    /// The Y component.
    pub y: f64,
}

impl Vec2 {
    /// Get the magnitude of the vector.
    #[must_use]
    pub fn mag(self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
    /// Convert this `Vec2` to a tuple of (usize, usize).
    #[must_use]
    pub fn to_usize(self) -> (usize, usize) {
        (self.x as usize, self.y as usize)
    }
    /// Convert this `Vec2` to a tuple of (isize, isize).
    #[must_use]
    pub fn to_isize(self) -> (isize, isize) {
        (self.x as isize, self.y as isize)
    }
    /// Get the distance between two points.
    #[must_use]
    pub fn dist(self, other: Self) -> f64 {
        (self - other).mag()
    }
}

impl From<(usize, usize)> for Vec2 {
    fn from(value: (usize, usize)) -> Self {
        Self {
            x: value.0 as f64,
            y: value.1 as f64,
        }
    }
}

impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

/// A 3D vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Vec3 {
    /// The X component.
    pub x: f64,
    /// The Y component.
    pub y: f64,
    /// The Z component.
    pub z: f64,
}

/// A quaternion.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Quaternion(pub f64, pub f64, pub f64, pub f64);

/// Multiply two quaternions.
#[must_use]
pub const fn quaternion_mul(q1: Quaternion, q2: Quaternion) -> Quaternion {
    Quaternion(
        q1.0 * q2.0 - q1.1 * q2.1 - q1.2 * q2.2 - q1.3 * q2.3,
        q1.0 * q2.0 + q1.1 * q2.1 - q1.2 * q2.2 + q1.3 * q2.3,
        q1.0 * q2.0 + q1.1 * q2.1 + q1.2 * q2.2 - q1.3 * q2.3,
        q1.0 * q2.0 - q1.1 * q2.1 + q1.2 * q2.2 + q1.3 * q2.3,
    )
}

/// Get the inverse of a quaternion.
#[must_use]
pub const fn quaternion_inv(q: Quaternion) -> Quaternion {
    Quaternion(q.0, -q.1, -q.2, -q.3)
}

/// Convert an angle-axis rotation to a quaternion.
#[must_use]
pub const fn aa_to_qua(aa: (f64, Vec3)) -> Quaternion {
    Quaternion(
        cos(aa.0 / 2.0),
        aa.1.x * sin(aa.0 / 2.0),
        aa.1.y * sin(aa.0 / 2.0),
        aa.1.z * sin(aa.0 / 2.0),
    )
}

/// Convert a Euler angle to a quaternion.
#[must_use]
pub const fn euler_to_qua(eul: Vec3) -> Quaternion {
    let roll = eul.z;
    let pitch = eul.y;
    let yaw = eul.z;

    Quaternion(
        cos(roll / 2.0) * cos(pitch / 2.0) * cos(yaw / 2.0)
            + sin(roll / 2.0) * sin(pitch / 2.0) * sin(yaw / 2.0),
        sin(roll / 2.0) * cos(pitch / 2.0) * cos(yaw / 2.0)
            - cos(roll / 2.0) * sin(pitch / 2.0) * sin(yaw / 2.0),
        cos(roll / 2.0) * sin(pitch / 2.0) * cos(yaw / 2.0)
            + sin(roll / 2.0) * cos(pitch / 2.0) * sin(yaw / 2.0),
        cos(roll / 2.0) * cos(pitch / 2.0) * sin(yaw / 2.0)
            + sin(roll / 2.0) * sin(pitch / 2.0) * cos(yaw / 2.0),
    )
}

impl Vec3 {
    /// Zeroed out `Vec3`
    pub const ZERO: Self = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    /// Get the magnitude of the vector.
    #[must_use]
    pub fn mag(self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
    /// Get the normalized version of the vector.
    #[must_use]
    pub fn normalized(self) -> Self {
        self / self.mag()
    }
    /// Get the distance between two Vec3s.
    #[must_use]
    pub fn dist(self, other: Self) -> f64 {
        (self - other).mag()
    }
    /// Rotate this vector around the origin. Provided rotation is in
    /// quaternion format.
    #[must_use]
    pub const fn rotate_around_origin_qua(self, rot: Quaternion) -> Self {
        let point_qua = Quaternion(0.0f64, self.x, self.y, self.z);

        let rot_inverse = quaternion_inv(rot);

        let out = quaternion_mul(quaternion_mul(rot_inverse, point_qua), rot);

        Self {
            x: out.1,
            y: out.2,
            z: out.3,
        }
    }
    /// Get the X and Y components of this `Vec3`.
    #[must_use]
    pub const fn xy(self) -> Vec2 {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }
    /// Linear interpolation.
    #[must_use]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        ((1.0 - t) * self) + (t * other)
    }
}

/// Struct for random sampling
pub struct Vec3Sampler {
    /// lower bound
    low: Vec3,
    /// higher bound
    high: Vec3,
}

impl UniformSampler for Vec3Sampler {
    type X = Vec3;

    fn new<B1, B2>(low: B1, high: B2) -> Result<Self, rand::distr::uniform::Error>
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Ok(Self {
            low: *low.borrow(),
            high: *high.borrow(),
        })
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Result<Self, rand::distr::uniform::Error>
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Ok(Self {
            low: *low.borrow(),
            high: (*high.borrow())
                + Vec3 {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                },
        })
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        Vec3 {
            x: rng.random_range(self.low.x..self.high.x),
            y: rng.random_range(self.low.y..self.high.y),
            z: rng.random_range(self.low.z..self.high.z),
        }
    }
}

impl SampleUniform for Vec3 {
    type Sampler = Vec3Sampler;
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3 {
            x: rhs.x * self,
            y: rhs.y * self,
            z: rhs.z * self,
        }
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// A point that has a color.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Point {
    /// The position.
    pub pos: Vec3,
    /// The color.
    pub color: Vec3,
}

/// The hidden trait for a scene type.
trait SceneTypeClosed: Clone + Copy + Debug + Default + PartialEq + Eq {}

/// A scene type.
#[expect(private_bounds, reason = "intentional")]
pub trait SceneType: SceneTypeClosed {
    /// The type of each item of a scene type.
    type Definition: AsRef<[Point]> + AsMut<[Point]> + for<'a> TryFrom<&'a [Point]> + Clone;
    /// The length of `Definition`.
    const NUM_PER_DEF: usize;
    /// The scene type variant.
    const SCENE_TYPE_VARIANT: SceneTypeEnum;
}

/// Enum for specifying a scene type in a match statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneTypeEnum {
    /// Points
    Point,
    /// Edges
    Edge,
    /// Tris
    Tri,
}

/// The point scene type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneTypePoint;

impl SceneTypeClosed for SceneTypePoint {}

impl SceneType for SceneTypePoint {
    type Definition = [Point; Self::NUM_PER_DEF];
    const NUM_PER_DEF: usize = 1;
    const SCENE_TYPE_VARIANT: SceneTypeEnum = SceneTypeEnum::Point;
}

/// The edge scene type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneTypeEdge;

impl SceneTypeClosed for SceneTypeEdge {}

impl SceneType for SceneTypeEdge {
    type Definition = [Point; Self::NUM_PER_DEF];
    const NUM_PER_DEF: usize = 2;
    const SCENE_TYPE_VARIANT: SceneTypeEnum = SceneTypeEnum::Edge;
}

/// The triangle scene type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneTypeTri;

impl SceneTypeClosed for SceneTypeTri {}

impl SceneType for SceneTypeTri {
    type Definition = [Point; Self::NUM_PER_DEF];
    const NUM_PER_DEF: usize = 3;
    const SCENE_TYPE_VARIANT: SceneTypeEnum = SceneTypeEnum::Tri;
}

/// A trait for rendering stuff.
pub trait Renderer {
    /// Render a color to the provided coordinate.
    fn render_screen_coord(&mut self, color: Vec3, coord: (usize, usize));
    /// Get the size of the screen.
    fn size(&self) -> (usize, usize);
}

/// A `Renderer` that renders to a buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct BufRenderer<const SIZE_X: usize, const SIZE_Y: usize> {
    /// A buffer.
    pub buf: Vec<Vec<Vec3>>,
}

impl<const SIZE_X: usize, const SIZE_Y: usize> Default for BufRenderer<SIZE_X, SIZE_Y> {
    fn default() -> Self {
        Self {
            buf: vec![vec![Vec3::default(); SIZE_X]; SIZE_Y],
        }
    }
}

impl<const SIZE_X: usize, const SIZE_Y: usize> Renderer for BufRenderer<SIZE_X, SIZE_Y> {
    fn render_screen_coord(&mut self, color: Vec3, coord: (usize, usize)) {
        if coord.0 < SIZE_X && coord.1 < SIZE_Y {
            self.buf[coord.1][coord.0] = color;
        }
    }
    fn size(&self) -> (usize, usize) {
        (SIZE_X, SIZE_Y)
    }
}

/// A scene.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scene<Type: SceneType> {
    /// The items of this scene.
    pub items: Vec<Box<Type::Definition>>,
}

impl<Type: SceneType> Scene<Type> {
    /// Create a new empty `Scene`.
    #[must_use]
    pub fn new_empty() -> Self {
        Self { items: Vec::new() }
    }
    /// Create a new `Scene` from the list of items. The length should be divisible
    /// by `Scene::Type::NUM_PER_DEF`.
    ///
    /// # Errors
    /// Returns an error if the list of provided items isn't dividible by `Scene::Type::NUM_PER_DEF`.
    #[expect(clippy::missing_panics_doc, reason = "false positive")]
    pub fn new_from_list(items: impl AsRef<[Point]>) -> anyhow::Result<Self> {
        let items = items.as_ref();
        ensure!(items.len() % Type::NUM_PER_DEF == 0, "Bad number of items");

        Ok(Self {
            items: items
                .chunks(Type::NUM_PER_DEF)
                .map(|v| Box::new(v.try_into().unwrap_or_else(|_| panic!("who broke math"))))
                .collect(),
        })
    }
    /// Render this scene to the provided renderer.
    pub fn render(&self, renderer: &mut dyn Renderer, camera_pos: Vec3, camera_rot: Vec3) {
        fn project(d: Vec3, e: Vec3) -> (usize, usize) {
            Vec2 {
                x: (e.z / d.z) * d.x + e.x,
                y: (e.z / d.z) * d.y + e.y,
            }
            .to_usize()
        }
        fn render_point(renderer: &mut dyn Renderer, out_point: (usize, usize), color: Vec3) {
            for x in out_point.0..=out_point.0 + 3 {
                for y in out_point.1..=out_point.1 + 3 {
                    renderer.render_screen_coord(color, (x, y));
                }
            }
        }
        let e = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 6.0,
        };

        let camera_rot_cos = Vec3 {
            x: camera_rot.x.cos(),
            y: camera_rot.y.cos(),
            z: camera_rot.z.cos(),
        };
        let camera_rot_sin = Vec3 {
            x: camera_rot.x.sin(),
            y: camera_rot.y.sin(),
            z: camera_rot.z.sin(),
        };

        let points: Vec<Vec<Point>> = self
            .items
            .iter()
            .map(|v| {
                v.as_ref()
                    .as_ref()
                    .iter()
                    .map(|v| {
                        let mut v = *v;
                        //v.pos = v.pos.rotate_around_origin_qua(camera_rot);
                        //v.pos -= camera_pos;
                        v.pos.x = camera_rot_cos.y
                            * (camera_rot_sin.z * (v.pos.y - camera_pos.y)
                                + camera_rot_cos.z * (v.pos.x - camera_pos.x))
                            - camera_rot_sin.y * (v.pos.z - camera_pos.z);

                        v.pos.y = camera_rot_sin.x
                            * (camera_rot_cos.z * (v.pos.z - camera_pos.z)
                                + camera_rot_sin.y
                                    * (camera_rot_sin.z * (v.pos.y - camera_pos.y)
                                        + camera_rot_cos.z * (v.pos.x - camera_pos.x)))
                            + camera_rot_cos.x
                                * (camera_rot_cos.y * (v.pos.y - camera_pos.y)
                                    - camera_rot_sin.z * (v.pos.x - camera_pos.x));

                        v.pos.z = camera_rot_cos.x
                            * (camera_rot_cos.z * (v.pos.z - camera_pos.z)
                                + camera_rot_sin.y
                                    * (camera_rot_sin.z * (v.pos.y - camera_pos.y)
                                        + camera_rot_cos.z * (v.pos.x - camera_pos.x)))
                            - camera_rot_sin.x
                                * (camera_rot_cos.y * (v.pos.y - camera_pos.y)
                                    - camera_rot_sin.z * (v.pos.x - camera_pos.x));
                        v
                    })
                    .filter(|v| v.pos.z >= 0.0)
                    .collect::<Vec<Point>>()
            })
            .collect::<Vec<_>>();

        match Type::SCENE_TYPE_VARIANT {
            SceneTypeEnum::Point => {
                let points = points.iter().flatten().collect::<Vec<&Point>>();

                for point in points {
                    render_point(renderer, project(point.pos, e), point.color);
                }
            }
            SceneTypeEnum::Edge => {
                for edge in points {
                    let point1 = edge[0];
                    let point2 = edge[1];

                    let pos1 = project(point1.pos, e);
                    let pos2 = project(point2.pos, e);

                    let max_dist = Vec2::from(pos1).dist(pos2.into());

                    render_point(renderer, pos1, point1.color);
                    render_point(renderer, pos2, point2.color);

                    for point in bresenham::Bresenham::new(
                        (pos1.0 as isize, pos1.1 as isize),
                        (pos2.0 as isize, pos2.1 as isize),
                    ) {
                        let color = point1.color.lerp(
                            point2.color,
                            Vec2 {
                                x: point.0 as f64,
                                y: point.1 as f64,
                            }
                            .dist(point1.pos.xy())
                                / max_dist,
                        );
                        renderer.render_screen_coord(color, (point.0 as usize, point.1 as usize));
                    }
                }
            }
            SceneTypeEnum::Tri => {}
        }
    }
}
