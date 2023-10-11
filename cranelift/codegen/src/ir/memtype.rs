//! Definitions for "memory types" in CLIF.
//!
//! A memory type is a struct-like definition -- fields with offsets,
//! each field having a type and possibly an attached fact -- that we
//! can use in proof-carrying code to validate accesses to structs and
//! propagate facts onto the loaded values as well.
//!
//! Memory types are meant to be rich enough to describe the *layout*
//! of values in memory, but do not necessarily need to embody
//! higher-level features such as subtyping directly. Rather, they
//! should encode an implementation of a type or object system.
//!
//! TODO:
//! - A memory type for discriminated unions (enums) to allow verified
//!   downcasting based on tags.
//! - A memory type for dynamic-length arrays.

use crate::ir::entities::MemoryType;
use crate::ir::pcc::Fact;
use crate::ir::Type;
use alloc::vec::Vec;
use cranelift_entity::PrimaryMap;

#[cfg(feature = "enable-serde")]
use serde_derive::{Deserialize, Serialize};

/// Data defining a memory type.
///
/// A memory type corresponds to a layout of data in memory. It may
/// have a statically-known or dynamically-known size.
#[derive(Clone, PartialEq, Hash)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub enum MemoryTypeData {
    /// An aggregate consisting of certain fields at certain offsets.
    ///
    /// Fields must be sorted by offset, and this is checked by the
    /// CLIF verifier. However, fields *may* overlap: this is to allow
    /// for enums, and to allow for the fact that individual fields
    /// may be dynamically sized.
    Struct {
        /// Size of this type.
        size: u64,

        /// Fields in this type. Sorted by offset.
        fields: Vec<MemoryTypeField>,
    },

    /// An aggregate consisting of a single element repeated at a
    /// certain stride, with a statically-known length (element
    /// count). Layout is assumed to be contiguous, with a stride
    /// equal to the element type's size (so, if the stride is greater
    /// than the struct's packed size, be sure to include padding in
    /// the memory-type definition).
    StaticArray {
        /// The element type. May be another array, a struct, etc.
        element: MemoryType,

        /// Number of elements.
        length: u64,
    },

    /// A single Cranelift primitive of the given type stored in
    /// memory.
    Primitive {
        /// The primitive type.
        ty: Type,
    },

    /// A type with no size.
    Empty,
}

impl std::default::Default for MemoryTypeData {
    fn default() -> Self {
        Self::Empty
    }
}

impl std::fmt::Display for MemoryTypeData {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Struct { size, fields } => {
                write!(f, "struct {size} {{")?;
                let mut first = true;
                for field in fields {
                    if first {
                        first = false;
                    } else {
                        write!(f, ",")?;
                    }
                    write!(f, " {}: {}", field.offset, field.ty)?;
                    if field.readonly {
                        write!(f, " readonly")?;
                    }
                    if let Some(fact) = &field.fact {
                        write!(f, " ! {}", fact)?;
                    }
                }
                write!(f, " }}")?;
                Ok(())
            }
            Self::StaticArray { element, length } => {
                write!(f, "static_array {element} * {length:#x}")
            }
            Self::Primitive { ty } => {
                write!(f, "primitive {ty}")
            }
            Self::Empty => {
                write!(f, "empty")
            }
        }
    }
}

/// One field in a memory type.
#[derive(Clone, PartialEq, Hash)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct MemoryTypeField {
    /// The offset of this field in the memory type.
    pub offset: u64,
    /// The type of the value in this field. Accesses to the field
    /// must use this type (i.e., cannot bitcast/type-pun in memory).
    pub ty: MemoryType,
    /// A proof-carrying-code fact about this value, if any.
    pub fact: Option<Fact>,
    /// Whether this field is read-only, i.e., stores should be
    /// disallowed.
    pub readonly: bool,
}

impl MemoryTypeData {
    /// Provide the static size of this type, if known.
    pub fn static_size(&self, tys: &PrimaryMap<MemoryType, MemoryTypeData>) -> Option<u64> {
        match self {
            Self::Struct { size, .. } => Some(*size),
            // Note that we disallow cyclic memtype references, so this must terminate.
            Self::StaticArray { element, length } => tys[*element]
                .static_size(tys)
                .and_then(|elt_size| elt_size.checked_mul(*length)),
            Self::Primitive { ty } => Some(ty.bytes() as u64),
            Self::Empty => Some(0),
        }
    }

    /// Visit all referenced memtypes in this memtype.
    pub fn visit_memtypes<F: FnMut(MemoryType)>(&self, mut f: F) {
        match self {
            Self::Struct { fields, .. } => {
                for field in fields {
                    f(field.ty);
                }
            }
            Self::StaticArray { element, .. } => f(*element),
            Self::Primitive { .. } | Self::Empty => {}
        }
    }
}
