//! A language-agnostic model of DWARF types.
//!
//! Types are built lazily from DWARF DIEs and interned in an arena owned
//! by [`DebugInfo`](crate::DebugInfo); they are referred to by
//! [`TypeId`].

use std::rc::Rc;

/// Index of a type in a [`DebugInfo`](crate::DebugInfo)'s type arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeId(pub(crate) usize);

/// Source language of a compilation unit (and hence of its types).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    C,
    Cpp,
    Rust,
    /// JavaScript compiled to Wasm by NightMonkey (a vendor
    /// `DW_LANG` code; DWARF has none for JavaScript).
    JavaScript,
    Other,
}

/// NightMonkey's `DW_LANG` code for JavaScript.
pub const DW_LANG_NIGHTMONKEY_JS: gimli::DwLang = gimli::DwLang(0xa53a);

impl Lang {
    pub(crate) fn from_dwarf(lang: gimli::DwLang) -> Lang {
        match lang {
            gimli::DW_LANG_C
            | gimli::DW_LANG_C89
            | gimli::DW_LANG_C99
            | gimli::DW_LANG_C11
            | gimli::DW_LANG_C17 => Lang::C,
            gimli::DW_LANG_C_plus_plus
            | gimli::DW_LANG_C_plus_plus_03
            | gimli::DW_LANG_C_plus_plus_11
            | gimli::DW_LANG_C_plus_plus_14
            | gimli::DW_LANG_C_plus_plus_17
            | gimli::DW_LANG_C_plus_plus_20 => Lang::Cpp,
            gimli::DW_LANG_Rust => Lang::Rust,
            DW_LANG_NIGHTMONKEY_JS => Lang::JavaScript,
            _ => Lang::Other,
        }
    }

    /// Whether this language uses C-family syntax for display.
    pub fn is_c_like(self) -> bool {
        !matches!(self, Lang::Rust)
    }
}

/// Encoding of a base (primitive) type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseEnc {
    Signed,
    Unsigned,
    SignedChar,
    UnsignedChar,
    /// A Unicode scalar value (e.g., Rust `char`).
    Utf,
    Float,
    Bool,
    Other,
}

/// Kind of a pointer-like type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtrKind {
    Pointer,
    Reference,
    RvalueReference,
}

/// Kind of a struct-like aggregate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructKind {
    Struct,
    Class,
    Union,
}

/// A type qualifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qual {
    Const,
    Volatile,
    Restrict,
    Atomic,
}

impl Qual {
    pub fn keyword(self) -> &'static str {
        match self {
            Qual::Const => "const",
            Qual::Volatile => "volatile",
            Qual::Restrict => "restrict",
            Qual::Atomic => "_Atomic",
        }
    }
}

/// A data member of a struct/union/variant.
#[derive(Clone, Debug)]
pub struct Member {
    /// Member name, if any (anonymous unions/structs have none).
    pub name: Option<String>,
    /// Member type.
    pub ty: TypeId,
    /// Offset from the start of the containing object, in bits.
    pub bit_offset: u64,
    /// For bitfields, the size in bits.
    pub bit_size: Option<u64>,
    /// Whether the member is compiler-generated.
    pub artificial: bool,
}

impl Member {
    /// Byte offset of this member (rounded down for bitfields).
    pub fn byte_offset(&self) -> u64 {
        self.bit_offset / 8
    }
}

/// One variant of a variant part (e.g., one Rust enum variant).
#[derive(Clone, Debug)]
pub struct Variant {
    /// The discriminant value selecting this variant; `None` for the
    /// default variant.
    pub discr_value: Option<u64>,
    /// Members of this variant (for Rust, a single member whose type is
    /// the variant's payload struct).
    pub members: Vec<Member>,
}

/// A DWARF variant part: a discriminant plus a set of variants.
#[derive(Clone, Debug)]
pub struct VariantPart {
    /// The discriminant member, if any.
    pub discr: Option<Member>,
    pub variants: Vec<Variant>,
}

/// A struct, class, or union.
#[derive(Clone, Debug)]
pub struct StructType {
    pub kind: StructKind,
    pub members: Vec<Member>,
    pub variant_part: Option<VariantPart>,
    /// Template (generic) type parameters, by name.
    pub template_params: Vec<(String, TypeId)>,
    /// Whether this is only a declaration (incomplete type).
    pub declaration: bool,
}

impl StructType {
    /// Whether this is a Rust-style enum: a struct whose only content is
    /// a variant part.
    pub fn is_enum_like(&self) -> bool {
        self.variant_part.is_some() && self.members.iter().all(|m| m.artificial)
    }

    /// Whether the members are tuple-like (`__0`, `__1`, ...).
    pub fn is_tuple_like(&self) -> bool {
        !self.members.is_empty()
            && self
                .members
                .iter()
                .enumerate()
                .all(|(i, m)| m.name.as_deref() == Some(&format!("__{i}")))
    }

    /// Find a template parameter by name.
    pub fn template_param(&self, name: &str) -> Option<TypeId> {
        self.template_params
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| *t)
    }
}

/// The structure of a type.
#[derive(Clone, Debug)]
pub enum TypeKind {
    Void,
    Base(BaseEnc),
    Pointer {
        pointee: Option<TypeId>,
        kind: PtrKind,
    },
    Struct(StructType),
    Enum {
        underlying: Option<TypeId>,
        enumerators: Vec<(String, i128)>,
    },
    Array {
        elem: TypeId,
        len: Option<u64>,
    },
    Typedef(Option<TypeId>),
    Qualified {
        qual: Qual,
        target: Option<TypeId>,
    },
    Function {
        ret: Option<TypeId>,
        params: Vec<TypeId>,
    },
    Unknown,
}

/// A type.
#[derive(Clone, Debug)]
pub struct Type {
    /// The (qualified, for Rust/C++) name of the type, if it has one.
    pub name: Option<String>,
    /// Size in bytes, if known.
    pub size: Option<u64>,
    pub kind: TypeKind,
    /// Language of the defining compilation unit.
    pub lang: Lang,
}

/// The type arena.
#[derive(Default)]
pub(crate) struct TypeArena {
    pub(crate) types: Vec<Rc<Type>>,
}

impl TypeArena {
    pub(crate) fn alloc(&mut self, ty: Type) -> TypeId {
        let id = TypeId(self.types.len());
        self.types.push(Rc::new(ty));
        id
    }
}

/// Make a Rust type name more readable by dropping default allocator /
/// hasher parameters.
pub fn simplify_rust_type_name(name: &str) -> String {
    name.replace(", alloc::alloc::Global", "")
        .replace(", std::hash::random::RandomState", "")
}

/// Make a C++ type name more readable by hiding libc++'s inline ABI
/// namespace (`std::__2::vector` becomes `std::vector`).
pub fn simplify_cpp_type_name(name: &str) -> String {
    name.replace("std::__2::", "std::")
        .replace("std::__1::", "std::")
}
