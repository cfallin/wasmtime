//! "Access path" expressions: a deliberately small language for
//! naming values, not a general expression evaluator.
//!
//! Grammar:
//!
//! ```text
//! path    := '*' path | '&' path | postfix
//! postfix := primary ( '.' field | '->' field | '[' int ']' )*
//! primary := ident ('::' ident)*     -- variable (local, then global)
//!          | '$' int | '$' | '$$' int? -- value history
//!          | '$' ident               -- a named Wasm entity (from the
//!                                       name section), e.g. `$x`
//!          | '%' ident               -- a built-in debugger name, e.g.
//!                                       `%local3`, `%pc`
//!          | int                     -- integer literal
//!          | '(' path ')'
//! field   := ident | int             -- `.0` selects a tuple field
//! ```

use crate::value::{Ctx, Value};
use anyhow::{Result, bail};

/// A reference into the value history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HistoryRef {
    /// `$N`: absolute history number.
    Absolute(usize),
    /// `$` (0), `$$` (1), `$$N` (N): relative to the most recent value.
    Relative(usize),
}

/// A parsed access path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Path {
    /// A source-level variable name (possibly `::`-qualified).
    Var(String),
    /// A value-history reference.
    History(HistoryRef),
    /// A `$name` reference to a Wasm entity by its name-section name
    /// (e.g., a local declared as `(local $x i32)`), interpreted by the
    /// debugger front-end.
    WasmName(String),
    /// A `%name` built-in debugger name (e.g., `%local3`, `%pc`),
    /// interpreted by the debugger front-end.
    Builtin(String),
    /// An integer literal.
    Int(u64),
    Deref(Box<Path>),
    AddrOf(Box<Path>),
    /// `.field` (auto-dereferences pointers).
    Field(Box<Path>, String),
    /// `->field`.
    Arrow(Box<Path>, String),
    Index(Box<Path>, u64),
}

impl Path {
    /// The root (innermost primary) of this path.
    pub fn root(&self) -> &Path {
        match self {
            Path::Deref(p)
            | Path::AddrOf(p)
            | Path::Field(p, _)
            | Path::Arrow(p, _)
            | Path::Index(p, _) => p.root(),
            p => p,
        }
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Path::Var(n) => write!(f, "{n}"),
            Path::History(HistoryRef::Absolute(n)) => write!(f, "${n}"),
            Path::History(HistoryRef::Relative(0)) => write!(f, "$"),
            Path::History(HistoryRef::Relative(1)) => write!(f, "$$"),
            Path::History(HistoryRef::Relative(n)) => write!(f, "$${n}"),
            Path::WasmName(n) => write!(f, "${n}"),
            Path::Builtin(n) => write!(f, "%{n}"),
            Path::Int(i) => write!(f, "{i}"),
            Path::Deref(p) => write!(f, "*{p}"),
            Path::AddrOf(p) => write!(f, "&{p}"),
            Path::Field(p, n) => write!(f, "{}.{n}", Paren(p)),
            Path::Arrow(p, n) => write!(f, "{}->{n}", Paren(p)),
            Path::Index(p, i) => write!(f, "{}[{i}]", Paren(p)),
        }
    }
}

/// Display helper that parenthesizes prefix operators under postfix
/// ones.
struct Paren<'a>(&'a Path);

impl std::fmt::Display for Paren<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Path::Deref(_) | Path::AddrOf(_) => write!(f, "({})", self.0),
            p => write!(f, "{p}"),
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
}

/// Parse an access path.
pub fn parse(input: &str) -> Result<Path> {
    let mut p = Parser {
        s: input.as_bytes(),
        pos: 0,
    };
    let path = p.path()?;
    p.ws();
    if p.pos != p.s.len() {
        bail!("unexpected `{}` in path expression", &input[p.pos..]);
    }
    Ok(path)
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.pos < self.s.len() && self.s[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.pos).copied()
    }

    fn eat(&mut self, tok: &str) -> bool {
        self.ws();
        if self.s[self.pos..].starts_with(tok.as_bytes()) {
            self.pos += tok.len();
            true
        } else {
            false
        }
    }

    fn path(&mut self) -> Result<Path> {
        if self.eat("*") {
            return Ok(Path::Deref(Box::new(self.path()?)));
        }
        if self.eat("&") {
            return Ok(Path::AddrOf(Box::new(self.path()?)));
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Path> {
        let mut p = self.primary()?;
        loop {
            if self.eat("->") {
                p = Path::Arrow(Box::new(p), self.field()?);
            } else if self.eat(".") {
                p = Path::Field(Box::new(p), self.field()?);
            } else if self.eat("[") {
                let i = self.int()?;
                if !self.eat("]") {
                    bail!("expected `]`");
                }
                p = Path::Index(Box::new(p), i);
            } else {
                return Ok(p);
            }
        }
    }

    fn field(&mut self) -> Result<String> {
        self.ws();
        let start = self.pos;
        while self.pos < self.s.len()
            && (self.s[self.pos].is_ascii_alphanumeric() || self.s[self.pos] == b'_')
        {
            self.pos += 1;
        }
        if start == self.pos {
            bail!("expected a field name");
        }
        Ok(String::from_utf8_lossy(&self.s[start..self.pos]).into_owned())
    }

    fn ident(&mut self) -> Option<String> {
        self.ws();
        let start = self.pos;
        loop {
            while self.pos < self.s.len()
                && (self.s[self.pos].is_ascii_alphanumeric() || self.s[self.pos] == b'_')
            {
                self.pos += 1;
            }
            // Allow `::` path separators.
            if self.s[self.pos..].starts_with(b"::")
                && self
                    .s
                    .get(self.pos + 2)
                    .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
            {
                self.pos += 2;
                continue;
            }
            break;
        }
        if start == self.pos {
            None
        } else {
            Some(String::from_utf8_lossy(&self.s[start..self.pos]).into_owned())
        }
    }

    fn int(&mut self) -> Result<u64> {
        self.ws();
        let start = self.pos;
        let (radix, digits_start) =
            if self.s[self.pos..].starts_with(b"0x") || self.s[self.pos..].starts_with(b"0X") {
                (16, self.pos + 2)
            } else {
                (10, self.pos)
            };
        self.pos = digits_start;
        while self.pos < self.s.len() && (self.s[self.pos] as char).is_digit(radix) {
            self.pos += 1;
        }
        if self.pos == digits_start {
            self.pos = start;
            bail!("expected an integer");
        }
        let text = std::str::from_utf8(&self.s[digits_start..self.pos]).unwrap();
        Ok(u64::from_str_radix(text, radix)?)
    }

    fn primary(&mut self) -> Result<Path> {
        match self.peek() {
            Some(b'(') => {
                self.pos += 1;
                let p = self.path()?;
                if !self.eat(")") {
                    bail!("expected `)`");
                }
                Ok(p)
            }
            Some(b'$') => {
                self.pos += 1;
                if self.s.get(self.pos) == Some(&b'$') {
                    self.pos += 1;
                    if self.s.get(self.pos).is_some_and(|c| c.is_ascii_digit()) {
                        let n = self.int()?;
                        return Ok(Path::History(HistoryRef::Relative(usize::try_from(n)?)));
                    }
                    return Ok(Path::History(HistoryRef::Relative(1)));
                }
                match self.s.get(self.pos) {
                    Some(c) if c.is_ascii_digit() => {
                        let n = self.int()?;
                        Ok(Path::History(HistoryRef::Absolute(usize::try_from(n)?)))
                    }
                    Some(c) if c.is_ascii_alphabetic() || *c == b'_' => {
                        let id = self.field()?;
                        Ok(Path::WasmName(id))
                    }
                    _ => Ok(Path::History(HistoryRef::Relative(0))),
                }
            }
            Some(b'%') => {
                self.pos += 1;
                Ok(Path::Builtin(self.field()?))
            }
            Some(c) if c.is_ascii_digit() => Ok(Path::Int(self.int()?)),
            Some(_) => match self.ident() {
                Some(id) => Ok(Path::Var(id)),
                None => bail!("expected a variable name"),
            },
            None => bail!("expected a path expression"),
        }
    }
}

/// Apply the non-root parts of `path` to a root value. `root` resolves
/// the path's root primary into a value.
pub fn eval(
    ctx: &Ctx<'_>,
    path: &Path,
    root: &mut dyn FnMut(&Path) -> Result<Value>,
) -> Result<Value> {
    match path {
        Path::Deref(p) => ctx.deref(&eval(ctx, p, root)?),
        Path::AddrOf(p) => ctx.address_of(&eval(ctx, p, root)?),
        Path::Field(p, f) => ctx.member(&eval(ctx, p, root)?, f),
        Path::Arrow(p, f) => {
            let v = eval(ctx, p, root)?;
            let d = ctx.deref(&v)?;
            ctx.member(&d, f)
        }
        Path::Index(p, i) => ctx.index(&eval(ctx, p, root)?, *i),
        p => root(p),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Path {
        parse(s).unwrap()
    }

    #[test]
    fn parse_paths() {
        assert_eq!(p("x"), Path::Var("x".into()));
        assert_eq!(
            p("p->x"),
            Path::Arrow(Box::new(Path::Var("p".into())), "x".into())
        );
        assert_eq!(
            p("arr[2].x"),
            Path::Field(
                Box::new(Path::Index(Box::new(Path::Var("arr".into())), 2)),
                "x".into()
            )
        );
        assert_eq!(p("*p"), Path::Deref(Box::new(Path::Var("p".into()))));
        assert_eq!(p("$1->x").to_string(), "$1->x");
        assert_eq!(p("$"), Path::History(HistoryRef::Relative(0)));
        assert_eq!(p("$$"), Path::History(HistoryRef::Relative(1)));
        assert_eq!(p("$$3"), Path::History(HistoryRef::Relative(3)));
        assert_eq!(p("$x"), Path::WasmName("x".into()));
        assert_eq!(p("$local3"), Path::WasmName("local3".into()));
        assert_eq!(p("%local3"), Path::Builtin("local3".into()));
        assert_eq!(p("%pc").to_string(), "%pc");
        assert_eq!(p("$p.2[1]").to_string(), "$p.2[1]");
        assert!(parse("%").is_err());
        assert_eq!(
            p("tuple.0"),
            Path::Field(Box::new(Path::Var("tuple".into())), "0".into())
        );
        assert_eq!(p("a::b::C"), Path::Var("a::b::C".into()));
        assert_eq!(p("arr[0x10]").to_string(), "arr[16]");
        assert_eq!(p("(*p).x").to_string(), "(*p).x");
        assert_eq!(p("*p.x").to_string(), "*p.x");
        assert_eq!(p("&s.x").to_string(), "&s.x");
        assert!(parse("a +").is_err());
        assert!(parse("a[").is_err());
    }
}
