//! Support for reference types (GC pointers) and safepoints for precise GC.
//!
//! Some values in the program dataflow are reference-typed (or reftyped for
//! short). These values have types `R32` or `R64`.
//!
//! To allow for precise GC, the codegen inserts "safepoints" at lowering. Every
//! instruction conceptually has a bit indicating whether it is a safepoint. If
//! an instruction is a safepoint, then no reftyped value may be live across the
//! instruction in a register. Instead, each such value must pass through memory
//! in a stack storage sloe. At emission time, the backend provides metadata to
//! the `CodeSink` that includes "stackmaps" at each safepoint. These stackmaps
//! provide a *precise* indication of which stack slots have reftyped values; no
//! reftyped value is missing (or else GC might miss a root) and no slot is
//! marked as reftyped that is not actually so (or else precise GC, and
//! especially moving/compacting GC, would be impossible).
//!
//! # Dataflow Analysis
//!
//! The basic idea here is to do some analysis before lowering from CLIF to
//! VCode, and then use these analysis results during lowering to insert the
//! necessary stores and loads (which are like spills and reloads, but not
//! managed by the register allocator). In essence, we implement the following
//! dataflow algorithm:
//!
//! - At every program point, for every reftyped SSA value, we track whether the
//!   value is "in register" or "on stack" or "uninit". These states form a
//!   lattice:
//!   
//! ```plain
//!
//!         Undef (top)
//!         /  ... \
//!    Slot(0) ... Slot(n)
//!      |           \
//!   Reg+Slot(0) ... Reg+Slot(n)
//!          \        /
//!             Reg
//! ```
//!
//!   The meaning of `Slot(i)` is that the given value's reference is
//!   actually stored in the given stack slot. (Reference slots have their own
//!   index space, separate from ordinary CLIF stackslots and regalloc
//!   spillslots.) To be used, it must be reloaded from the slot.
//!
//!   The meaning of `Reg+Slot(i)` is that the value is in both the SSA value
//!   itself and also in the given slot. The value in the slot is *up-to-date*,
//!   i.e., is equal to the SSA value.
//!
//!   The meaning of `Reg` is that the current value is only in the value
//!   itself (or more precisely, the virtual register assigned to it during
//!   lowering).
//!
//!   This lattice has a meet function defined in the usual way, given the
//!   partial order shown above: Undef > Slot(i), Slot(i) > Reg+Slot(i),
//!   Reg+Slot(i) > Reg.
//!
//! - At a *use* of `V`, the transfer function (for the analysis value attached
//!   to `V`) works as follows:
//!
//!     Undef => error
//!     Slot(i) => Reg+Slot(i)     , emit load(Slot(i)) before use
//!     Reg+Slot(i) => Reg+Slot(i)
//!     Reg => Reg
//!
//! - At a *def* of `V`, the resulting analysis value in the post-state for `V`
//!   is always `Reg`.
//!
//! - At *safepoints*, we transform the state as follows:
//!
//!     for each V in pre-state:
//!
//!       Undef => Undef
//!       Slot(i) => Slot(i)
//!       Reg+Slot(i) => Slot(i)
//!       Reg => Slot(i')        , where i' is newly allocated;
//!                                emit store(Slot(i') before safepoint
//!
//! - At *meets*, if any input I is > Reg+Slot (i.e., is not in a register),
//!   and output is Reg+Slot or Reg, handle as a use and emit load as necessary.
//!
//! # Implementation
//!
//! To implement the above, we perform a standard dataflow analysis using a
//! workqueue algorithm, computing the analysis state (mapping from vregs to
//! analysis values) at each block entry until fixpoint then stepping through
//! each block with the transfer function to find the value at each program
//! point.
//!
//! We take as input the `BlockLoweringOrder` so that we can perform the
//! analysis in terms of final (`MachInst`) blocks. This also provides us an
//! edge at which to insert any needed loads for meets.
//!
//! The result of this analysis is:
//!
//! - The number of reference slots needed.
//! - A set of loads and stores from/to reference slots at certain program
//!   points (before instructions, and on block edges).
//! - For each instruction that is a safepoint, a list of reference slots that
//!   contain active references.

use crate::inst_predicates::is_safepoint;
use crate::ir::{Block, Function, Inst};
use crate::fx::FxHashMap;
use crate::entity::SecondaryMap;
use crate::ir::types::{R32, R64};

/// A reference slot, used to store a reference during a safepoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RefSlot(u32);

impl RefSlot {
    /// Get the refslot number.
    pub fn get(self) -> u32 {
        self.0
    }

    fn next(self) -> RefSlot {
        RefSlot(self.0 + 1)
    }
}

/// An analysis value: for a given vreg, where does its value actually live at a
/// given program point? `Slot` indicates it is stored in a reference slot, so
/// it can be observed by a safepoint; `SlotAndReg` indicates both the actual
/// vreg and a slot, and that the slot value is up-to-date; and `Reg` indicates
/// vreg-only. `Error` indicates that a ref and non-ref value have met, which is
/// impossible in a system that supports precise GC / reference-tracking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AnalysisValue {
    Undef,
    Slot(RefSlot),
    SlotAndReg(RefSlot),
    Reg,
    Error,
}

/// A "fixup inst": a load or store to move values between vregs and slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixupInst {
    /// A load of the vreg form the given refslot.
    Load(RefSlot),
    /// A store of the vreg to the given refslot.
    Store(RefSlot),
}

impl AnalysisValue {
    /// Meet one analysis value with another, producing fixup insts that should
    /// be generated on one in-edge or the other to produce the meet.
    fn meet(self, other: AnalysisValue) -> (AnalysisValue, Option<FixupInst>, Option<FixupInst>) {
        match (self, other) {
            // Two equal values meet to themselves, with no fixup.
            (a, b) if a == b => (a, None, None),
            // Error is sticky.
            (AnalysisValue::Error, _) | (_, AnalysisValue::Error) => {
                (AnalysisValue::Error, None, None)
            }
            // An undef meeting any other value results in imprecise reference
            // tracking (the result may or may not be a reference).
            (AnalysisValue::Undef, _) | (_, AnalysisValue::Undef) => {
                (AnalysisValue::Error, None, None)
            }

            // Case analysis: now have only `Reg`, `SlotAndReg`, and `Slot`
            // left.

            // Differing slot numbers force to Reg only.
            (AnalysisValue::Slot(slot1), AnalysisValue::Slot(slot2))
            | (AnalysisValue::Slot(slot1), AnalysisValue::SlotAndReg(slot2))
            | (AnalysisValue::SlotAndReg(slot1), AnalysisValue::Slot(slot2))
            | (AnalysisValue::SlotAndReg(slot1), AnalysisValue::SlotAndReg(slot2))
                if slot1 != slot2 =>
            {
                (AnalysisValue::Reg, None, None)
            }

            // Case analysis: now have only Reg-SlotAndReg, Reg-Slot, and
            // Slot-SlotAndReg left, all with equal slot numbers. Reflexive (x
            // meet x) case handled above, and same kind differing slots handled
            // already too.

            // Slot merges with SlotAndReg (same slot) to produce SlotAndReg,
            // with a fixup load on the Slot side.
            (AnalysisValue::Slot(slot1), AnalysisValue::SlotAndReg(slot2)) => {
                debug_assert_eq!(slot1, slot2);
                (
                    AnalysisValue::SlotAndReg(slot1),
                    Some(FixupInst::Load(slot1)),
                    None,
                )
            }
            (AnalysisValue::SlotAndReg(slot1), AnalysisValue::Slot(slot2)) => {
                debug_assert_eq!(slot1, slot2);
                (
                    AnalysisValue::SlotAndReg(slot1),
                    None,
                    Some(FixupInst::Load(slot2)),
                )
            }

            // Slot or SlotAndReg merge with Reg to become just Reg. (At this
            // point, Undef and Error are handled already, so we use `_`.)
            (AnalysisValue::Reg, _) | (_, AnalysisValue::Reg) => (AnalysisValue::Reg, None, None),

            _ => unreachable!(),
        }
    }

    /// Handle a use, transitioning the analysis value and optionally emitting a
    /// fixup before the insn.
    fn with_use(&mut self) -> Option<FixupInst> {
        match *self {
            // Undef or error: handled elsewhere.
            AnalysisValue::Undef | AnalysisValue::Error => panic!("Use of undef or error value"),

            // Slot only: must load into reg.
            AnalysisValue::Slot(slot) => {
                *self = AnalysisValue::SlotAndReg(slot);
                Some(FixupInst::Load(slot))
            }

            // Slot-and-reg or reg-only: no action needed.
            AnalysisValue::SlotAndReg(_) | AnalysisValue::Reg => None,
        }
    }

    /// Handle a safepoint, optionally emitting a fixup and optionally adding a
    /// slot to the safepoint slot-set. Requires access to the next-slot
    /// counter.
    fn at_safepoint(&mut self, next_slot: &mut RefSlot) -> (Option<FixupInst>, Option<RefSlot>) {
        match *self {
            // Error: handled elsewhere.
            AnalysisValue::Error => panic!("Error value at safepoint"),

            // Undef: no ref to handle, no action rquired.
            AnalysisValue::Undef => (None, None),

            // Reg only: store to slot; value in slot may be different after
            // safepoint, so transition to `Slot(n)` only, not `SlotAndReg(n)`.
            AnalysisValue::Reg => {
                // Allocate a new slot number.
                let slot = *next_slot;
                *next_slot = slot.next();
                *self = AnalysisValue::Slot(slot);
                (Some(FixupInst::Store(slot)), Some(slot))
            }

            // SlotAndReg: transition to Slot only; provide slot to safepoint
            // set.
            AnalysisValue::SlotAndReg(slot) => {
                *self = AnalysisValue::Slot(slot);
                (None, Some(slot))
            }

            // Slot: remain in this state; provide slot to safepoint set.
            AnalysisValue::Slot(slot) => {
                (None, Some(slot))
            }
        }
    }
}

/// Analysis state at a program point: map from SSA values to `AnalysisValue`s.
#[derive(Debug, Clone)]
struct AnalysisState {
    values: FxHashMap<Value, AnalysisValue>,
}

impl AnalysisState {
    fn new() -> Self {
        AnalysisState {
            values: FxHashMap::default(),
        }
    }

    fn step_param_def(&mut self, func: &Function, val: Value) {
        if is_ref_typed(func, val) {
            self.values.insert(val, AnalysisValue::Reg);
        }
    }

    /// Step the AnalysisState over an instruction. Returns the fixups that must
    /// occur prior to this instruction, and the safepoint slots for any
    /// safepoint.
    fn step_insn(&mut self, func: &Function, inst: Inst) -> (Vec<FixupInst>, Vec<RefSlot>) {
        let mut fixups = vec![];
        let mut slots = vec![];
        for val in func.dfg.inst_args(inst) {
            let val = func.dfg.resolve_aliases(*val);
            if is_ref_typed(func, val) {
                if let Some(v) = self.values.get_mut(&val) {
                    if let Some(f) = v.with_use() {
                        fixups.push(f);
                    }
                }
            }
        }
        if is_safepoint(func, inst) {
            for v in self.values.values_mut() {
                let (fixup, slot) = v.at_safepoint();
                if fixup.is_some() {
                    fixups.push(fixup.unwrap());
                }
                if slot.is_some() {
                    slots.push(slot.unwrap());
                }
            }
        }
        for val in func.dfg.inst_results(inst) {
            self.values.insert(val, AnalysisValue::Reg);
        }
        ret
    }

    fn meet(&mut self, other: &AnalysisState) -> (Vec<FixupInst>, Vec<FixupInst>) {
    }

    /// Meets N analysis states at a meet point, providing the fixup
    /// instructions needed at each.
    fn meet_all(states: &[AnaysisState]) -> (AnalysisState, Vec<Vec<FixupInst>>) {
        if states.len() == 0 {
            return (AnalysisState::new(), vec![]);
        }

        let mut fixups = Vec::with_capacity(states.len());
        let mut merged = states[0].clone();
        for i in 1..states.len() {

    }
}

/// Safepoint analysis: for a given function, determine how many refslots are
/// needed, where loads and stores from/to the refslots must occur, and which
/// refslots contain references at each safepoint.
#[derive(Debug)]
pub struct SafepointAnalysis {
    pre_inst_fixups: SecondaryMap<Inst, Option<FixupInst>>,
    edge_fixups: FxHashMap<(Block, Block), Vec<FixupInst>>,
    safepoint_slots: FxHashMap<Inst, Vec<RefSlot>>,
}

fn is_ref_typed(func: &Function, val: Value) -> bool {
    let ty = func.dfg.value_type(val);
    ty == R32 || ty == R64
}

impl SafepointAnalysis {
    /// Compute an analysis result for a function.
    pub fn new(func: &Funcion) -> Self {
        let mut result = SafepointAnalysis {
            pre_inst_fixups: SecondaryMap::new(),
            edge_fixups: FxHashMap::default(),
            safepoint_slots: FxHashMap::default(),
        };

        let mut bb_start_state: SecondaryMap<Block, FxHashMap<Value, AnalysisValue>> = SecondaryMap::new();
        let mut 

        result
    }
}
