use std::collections::HashMap;

#[derive(Debug)]
pub enum Shape {
    Circle(f64),
    Rect { w: u32, h: u32 },
    Empty,
}

#[derive(Debug, Clone, Copy)]
pub enum Color {
    Red,
    Green,
    Blue,
}

#[derive(Debug)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

pub struct Wrapper(pub u64);

pub struct Node {
    pub value: i32,
    pub next: Option<Box<Node>>,
}

static GREETING: &str = "hello, world";

#[inline(never)]
fn describe(shape: &Shape, color: Color, point: &Point) -> String {
    let s = format!("{shape:?} {color:?} {point:?}");
    s
}

fn main() {
    let name = String::from("wasmtime");
    let nums: Vec<i32> = vec![1, 2, 3, 4];
    let words = vec!["a".to_string(), "bc".to_string()];
    let shapes = vec![
        Shape::Circle(1.5),
        Shape::Rect { w: 3, h: 4 },
        Shape::Empty,
    ];
    let maybe: Option<i32> = Some(7);
    let nothing: Option<i32> = None;
    let res: Result<u32, String> = Err("bad".to_string());
    let boxed = Box::new(Point { x: 5, y: -6 });
    let list = Node {
        value: 1,
        next: Some(Box::new(Node {
            value: 2,
            next: None,
        })),
    };
    let w = Wrapper(99);
    let tuple = (1u8, 'x', true);
    let slice: &[i32] = &nums[1..3];
    let mut map = HashMap::new();
    map.insert("one".to_string(), 1);
    map.insert("two".to_string(), 2);
    let color = Color::Blue;
    let point = Point { x: 1, y: 2 };
    let d = describe(&shapes[1], color, &point);
    println!(
        "{name} {nums:?} {words:?} {maybe:?} {nothing:?} {res:?} {} {} {} {:?} {slice:?} {} {d} {GREETING}",
        boxed.x,
        list.next.as_ref().unwrap().value,
        w.0,
        tuple,
        map.len()
    );
}
