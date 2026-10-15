//! Symbol demangling for Rust (legacy and v0 schemes) and C++
//! (Itanium ABI) names.

/// Demangle `name` if it looks like a mangled Rust or C++ symbol;
/// otherwise return it unchanged.
///
/// Rust hashes (e.g. `::h0123456789abcdef` or `[0123abcd]` crate
/// disambiguators) are omitted for readability.
pub fn demangle(name: &str) -> String {
    if let Ok(d) = rustc_demangle::try_demangle(name) {
        return format!("{d:#}");
    }
    if name.starts_with("_Z") {
        if let Ok(sym) = cpp_demangle::Symbol::new(name) {
            if let Ok(s) = sym.demangle() {
                return s;
            }
        }
    }
    name.to_string()
}

/// Return the "simple" (last path component) of a demangled name,
/// ignoring any generic arguments or C++ parameter list. For example,
/// `alloc::vec::Vec<T>::push` yields `push` and `ns::f(int)` yields `f`.
pub fn simple_name(name: &str) -> &str {
    // Drop a trailing C++ parameter list.
    let mut base = name;
    if base.ends_with(')') {
        let mut depth = 0;
        for (i, c) in base.char_indices().rev() {
            match c {
                ')' => depth += 1,
                '(' => {
                    depth -= 1;
                    if depth == 0 {
                        base = &base[..i];
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    // Find the last `::` outside of generic brackets, and the start of
    // any trailing generic arguments.
    let bytes = base.as_bytes();
    let mut depth = 0i32;
    let mut start = 0;
    let mut end = base.len();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => {
                if depth == 0 {
                    end = i;
                }
                depth += 1;
            }
            // `->` (in `fn() -> T`) is not a closing bracket.
            b'>' if i > 0 && bytes[i - 1] == b'-' => {}
            b'>' => depth -= 1,
            b':' if depth == 0 && bytes.get(i + 1) == Some(&b':') => {
                start = i + 2;
                end = base.len();
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start > end {
        return base;
    }
    &base[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_v0() {
        assert_eq!(
            demangle("_RNvCsdN1slkHwP8f_14debug_tui_rust4main"),
            "debug_tui_rust::main"
        );
    }

    #[test]
    fn rust_legacy() {
        assert_eq!(
            demangle("_ZN4core3fmt5write17h0123456789abcdefE"),
            "core::fmt::write"
        );
    }

    #[test]
    fn cpp() {
        assert_eq!(demangle("_ZN2ns3fooEi"), "ns::foo(int)");
    }

    #[test]
    fn passthrough() {
        assert_eq!(demangle("main"), "main");
    }

    #[test]
    fn simple() {
        assert_eq!(simple_name("alloc::vec::Vec<T>::push"), "push");
        assert_eq!(simple_name("ns::foo(int)"), "foo");
        assert_eq!(simple_name("main"), "main");
        assert_eq!(simple_name("core::option::Option<i32>"), "Option");
        assert_eq!(simple_name("<T as core::fmt::Debug>::fmt"), "fmt");
        assert_eq!(simple_name("Vec<a::b::C>"), "Vec");
        assert_eq!(
            simple_name("std::sync::LazyLock<u32, fn() -> u32>"),
            "LazyLock"
        );
    }
}
