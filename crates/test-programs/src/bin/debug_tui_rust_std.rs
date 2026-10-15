//! Exercises pretty-printing of Rust standard-library types in the
//! `--debug-tui` debugger.

use std::borrow::Cow;
use std::cell::{Cell, OnceCell, RefCell, UnsafeCell};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, LinkedList, VecDeque};
use std::mem::ManuallyDrop;
use std::num::{NonZeroU32, Wrapping};
use std::ops::Range;
use std::cell::LazyCell;
use std::fmt::Debug;
use std::rc::{Rc, Weak};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, RwLock};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Token {
    Num(i64),
    Ident(String),
    Pair { a: u8, b: u8 },
    Eof,
}

#[derive(Debug)]
pub struct Inner {
    pub id: u32,
    pub label: &'static str,
}

pub struct MyCustomPointer {
    pub _inner: Box<Vec<i32>>,
}

pub struct S<'a> {
    pub ptr: &'a i32,
    pub name: &'a str,
}

static GLOBAL_LAZY: LazyLock<Vec<u32>> = LazyLock::new(|| vec![7, 8, 9]);
static GLOBAL_LAZY_UNFORCED: LazyLock<u32> = LazyLock::new(|| 1);

#[inline(never)]
fn consume<T: ?Sized>(_t: &T) {}

fn main() {
    let vec: Vec<u32> = vec![10, 20, 30];
    let mut deque: VecDeque<i32> = VecDeque::with_capacity(4);
    // Make the ring buffer wrap around.
    deque.push_back(1);
    deque.push_back(2);
    deque.push_back(3);
    deque.pop_front();
    deque.pop_front();
    deque.push_back(4);
    deque.push_back(5);
    deque.push_front(0);
    let mut hash_map: HashMap<String, u32> = HashMap::new();
    hash_map.insert("a".to_string(), 1);
    hash_map.insert("b".to_string(), 2);
    let mut btree_map: BTreeMap<u32, &str> = BTreeMap::new();
    for i in 0..20 {
        btree_map.insert(i, if i % 2 == 0 { "even" } else { "odd" });
    }
    let hash_set: HashSet<u8> = [7u8, 8, 9].into_iter().collect();
    let btree_set: BTreeSet<char> = ['x', 'y', 'z'].into_iter().collect();
    let tokens = vec![
        Token::Num(-5),
        Token::Ident("foo".to_string()),
        Token::Pair { a: 1, b: 2 },
        Token::Eof,
    ];
    let string = String::from("hello");
    let cow_borrowed: Cow<'_, str> = Cow::Borrowed("borrowed");
    let cow_owned: Cow<'_, str> = Cow::Owned("owned".to_string());
    let cow_slice: Cow<'_, [u32]> = Cow::Borrowed(&[1, 2, 3]);
    let boxed: Box<Inner> = Box::new(Inner { id: 1, label: "boxed" });
    let boxed_slice: Box<[u16]> = vec![4u16, 5, 6].into_boxed_slice();
    let boxed_str: Box<str> = "boxed str".into();
    let mutex = Mutex::new(Inner { id: 2, label: "mutex" });
    let rwlock = RwLock::new(vec![1u8, 2]);
    let arc = Arc::new(Inner { id: 3, label: "arc" });
    let arc2 = arc.clone();
    let rc = Rc::new(String::from("rc"));
    let weak: Weak<String> = Rc::downgrade(&rc);
    let refcell = RefCell::new(vec![1i8, -1]);
    let cell = Cell::new(42u64);
    let unsafe_cell = UnsafeCell::new(7i16);
    let once_cell: OnceCell<u32> = OnceCell::new();
    once_cell.set(99).unwrap();
    let option_box: Option<Box<Inner>> = Some(Box::new(Inner { id: 4, label: "opt" }));
    let result: Result<Vec<u8>, String> = Ok(vec![1]);
    let manually_drop = ManuallyDrop::new(Inner { id: 5, label: "md" });
    let wrapping = Wrapping(250u8);
    let non_zero = NonZeroU32::new(17).unwrap();
    let range: Range<u32> = 3..9;
    let linked_list: LinkedList<u32> = [100, 200, 300].into_iter().collect();
    let binary_heap: BinaryHeap<u32> = [5, 1, 8].into_iter().collect();
    let arc_mutex = Arc::new(Mutex::new(HashMap::from([(1u8, 'a')])));
    let nested: Vec<Option<Rc<RefCell<Token>>>> =
        vec![Some(Rc::new(RefCell::new(Token::Num(3)))), None];
    let array = [1u8, 2, 3];
    let tuple = (1u8, "two", 3.0f32);
    let once_lock: OnceLock<String> = OnceLock::new();
    once_lock.set("set once".to_string()).unwrap();
    let once_lock_empty: OnceLock<u8> = OnceLock::new();
    let lazy_cell: LazyCell<u32> = LazyCell::new(|| 40 + 2);
    let lazy_cell_unforced: LazyCell<u32> = LazyCell::new(|| 0);
    LazyCell::force(&lazy_cell);
    LazyLock::force(&GLOBAL_LAZY);
    let guard_mutex = Mutex::new(vec![3u8, 4]);
    let mutex_guard = guard_mutex.lock().unwrap();
    let guard_rwlock = RwLock::new(5u16);
    let read_guard = guard_rwlock.read().unwrap();
    let guard_refcell = RefCell::new(Token::Ident("cell".to_string()));
    let ref_guard = guard_refcell.borrow();
    let arc_str: Arc<str> = Arc::from("arc str");
    let rc_slice: Rc<[u32]> = Rc::from(vec![11u32, 12]);
    let rc_str: Rc<str> = Rc::from("rc str");
    let boxed_dyn: Box<dyn Debug> = Box::new(Inner { id: 6, label: "dyn" });
    let ref_dyn: &dyn Debug = &tuple;
    let custom = MyCustomPointer {
        _inner: Box::new(vec![1, 2, 3]),
    };
    let x = 123;
    let s = S { ptr: &x, name: "s" };

    consume(&vec); // break here
    consume(&deque);
    consume(&hash_map);
    consume(&btree_map);
    consume(&hash_set);
    consume(&btree_set);
    consume(&tokens);
    consume(&string);
    consume(&cow_borrowed);
    consume(&cow_owned);
    consume(&cow_slice);
    consume(&boxed);
    consume(&boxed_slice);
    consume(&boxed_str);
    consume(&mutex);
    consume(&rwlock);
    consume(&arc2);
    consume(&weak);
    consume(&refcell);
    consume(&cell);
    consume(&unsafe_cell);
    consume(&once_cell);
    consume(&option_box);
    consume(&result);
    consume(&manually_drop);
    consume(&wrapping);
    consume(&non_zero);
    consume(&range);
    consume(&linked_list);
    consume(&binary_heap);
    consume(&arc_mutex);
    consume(&nested);
    consume(&array);
    consume(&tuple);
    consume(&once_lock);
    consume(&once_lock_empty);
    consume(&lazy_cell);
    consume(&lazy_cell_unforced);
    consume(&*mutex_guard);
    consume(&*read_guard);
    consume(&*ref_guard);
    consume(&arc_str);
    consume(&rc_slice);
    consume(&rc_str);
    consume(&boxed_dyn);
    consume(ref_dyn);
    consume(&custom);
    consume(&s);
    consume(&GLOBAL_LAZY_UNFORCED);
    println!("{vec:?} {deque:?} {} {}", btree_map.len(), linked_list.len());
}
