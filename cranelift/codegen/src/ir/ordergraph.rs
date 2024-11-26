//! Order graph: data structure used for PCC facts. Represents
//! inequalities between symbolic and concrete quantities.
//!
//! A node in the graph is an equivalence class of expressions. Each
//! expression is (i) a constant value, or (ii) a symbolic
//! var. Symbolic vars are allocated fresh as needed and correspond to
//! values in facts, but are not otherwise defined by or implicitly
//! equal to any value in the program; they are a separate index
//! space.
//!
//! A directed edge in the graph is either an equality edge or an
//! inequality edge (i.e., `==` or `<=`), with an optional
//! non-negative (zero or positive) offset.
//!
//! We support adding nodes, adding edges, union'ing nodes (equivalent
//! to adding an equality edge with zero offset), and
//! "simplification".
//!
//! The graph is acyclic once simplified: a cycle implies equality (if
//! all offsets are zero) or an impossible situation (if any offset is
//! greater than zero). In the latter case, simplification flags the
//! error and further use of the order graph is invalid.
//!
//! ## Simplification
//!
//! The goal of simplification is not to compute a transitive closure
//! (that would result in many inequality edges implied by transitive
//! combinations of existing edges), but rather to union as many nodes
//! together as possible according to equalities. We do this by, for
//! each node, iterating until fixpoint:
//!
//! - Sort all outgoing edges by destination node. If any two edges
//!   are to the same node, pick the edge that implies greater
//!   distance (so if x + 5 <= y and x + 10 <= y, then keep x + 10 <=
//!   y). Pick equality edges over inequality edges. If an equality
//!   edge contradicts an inequality (e.g., x + 5 <= y and x + 3 ==
//!   y), flag an error. If two equality edges contradict (e.g., x + 5
//!   == y and x + 3 == y), flag an error.
//! - If equality edges exist with the same offset to two distinct
//!   destination nodes, merge those two nodes into one.
//!
//! ## Queries
//!
//! We support one kind of query: is one value (constant or symbolic
//! value) necessarily less than or equal to another value, with at
//! least some offset? We compute the result of this query by
//! exploring outward.

use std::collections::hash_map::Entry;

use crate::entity::{entity_impl, PrimaryMap};
use crate::unionfind::UnionFind;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use rustc_hash::FxHashMap;

/// A node in the order graph.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrderNode(u32);
/// A variable in the order graph.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrderVar(u32);

entity_impl!(OrderNode, "order_node");
entity_impl!(OrderVar, "order_var");

/// An order graph.
#[derive(Clone, Debug, Default)]
pub struct OrderGraph {
    /// Nodes in the order graph.
    pub nodes: PrimaryMap<OrderNode, OrderNodeData>,
    /// Edges in the order graph.
    pub edges: BTreeSet<OrderEdge>,
    /// Next fresh variable to allocate.
    next_var: u32,
}

/// Data at one node in the order graph.
#[derive(Clone, Debug)]
pub struct OrderNodeData {
    /// Constant bound (equal or upper bound), if any.
    pub bound: ConstantBound,
    /// Variables in the equivalence class represented by this node.
    pub exprs: Vec<OrderVar>,
}

/// A constant-valued bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstantBound {
    /// No constant bound known.
    None,
    /// Equal to the given value.
    Eq(u64),
    /// Less than or equal to the given value.
    Le(u64),
}

impl ConstantBound {
    fn merge(a: ConstantBound, b: ConstantBound) -> Option<ConstantBound> {
        match (a, b) {
            (ConstantBound::None, b) => Some(b),
            (a, ConstantBound::None) => Some(a),
            (ConstantBound::Eq(a), ConstantBound::Eq(b)) if a == b => Some(ConstantBound::Eq(a)),
            (ConstantBound::Le(a), ConstantBound::Eq(b))
            | (ConstantBound::Eq(b), ConstantBound::Le(a))
                if b <= a =>
            {
                Some(ConstantBound::Eq(b))
            }
            (ConstantBound::Le(a), ConstantBound::Le(b)) => {
                let x = std::cmp::min(a, b);
                if x == 0 {
                    Some(ConstantBound::Eq(0))
                } else {
                    Some(ConstantBound::Le(x))
                }
            }
            _ => None,
        }
    }
}

/// An ordering edge from one node to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrderEdge {
    /// The left-hand-side node in the ordering relation.
    pub lhs: OrderNode,
    /// The right-hand side node in the ordering relation.
    pub rhs: OrderNode,
    /// Kind of edge: equality or inequality.
    pub kind: OrderEdgeKind,
    /// The offset (distance) denoted by this edge.
    pub offset: u64,
}

impl OrderEdge {
    fn compose_left(lhs_bound: &ConstantBound, edge: &OrderEdge) -> ConstantBound {
        match (lhs_bound, edge.kind) {
            (ConstantBound::None, _) => ConstantBound::None,
            (_, OrderEdgeKind::Le) => ConstantBound::None,
            (ConstantBound::Eq(x), OrderEdgeKind::Eq) => {
                ConstantBound::Eq(x.saturating_add(edge.offset))
            }
            (ConstantBound::Le(x), OrderEdgeKind::Eq) => {
                ConstantBound::Le(x.saturating_add(edge.offset))
            }
        }
    }
    fn compose_right(edge: &OrderEdge, rhs_bound: &ConstantBound) -> ConstantBound {
        match (edge.kind, rhs_bound) {
            (_, ConstantBound::None) => ConstantBound::None,
            (OrderEdgeKind::Eq, ConstantBound::Eq(x)) => {
                ConstantBound::Eq(x.saturating_sub(edge.offset))
            }
            (OrderEdgeKind::Eq, ConstantBound::Le(x))
            | (OrderEdgeKind::Le, ConstantBound::Eq(x))
            | (OrderEdgeKind::Le, ConstantBound::Le(x)) => {
                ConstantBound::Le(x.saturating_sub(edge.offset))
            }
        }
    }
    fn merge(a: &OrderEdge, b: &OrderEdge) -> Option<OrderEdge> {
        assert_eq!(a.lhs, b.lhs);
        assert_eq!(a.rhs, b.rhs);
        match (a.kind, b.kind) {
            (OrderEdgeKind::Eq, OrderEdgeKind::Eq) if a.offset == b.offset => Some(a.clone()),
            (OrderEdgeKind::Le, OrderEdgeKind::Le) => Some(OrderEdge {
                lhs: a.lhs,
                rhs: a.rhs,
                kind: OrderEdgeKind::Le,
                offset: std::cmp::max(a.offset, b.offset),
            }),
            (OrderEdgeKind::Le, OrderEdgeKind::Eq) if b.offset >= a.offset => Some(b.clone()),
            (OrderEdgeKind::Eq, OrderEdgeKind::Le) if a.offset >= b.offset => Some(a.clone()),
            _ => None,
        }
    }
}

/// An edge kind: equal or less-than-or-equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OrderEdgeKind {
    /// Equal.
    Eq,
    /// Less-than-or-equal.
    Le,
}

/// An error (inconsistency/contradiction) discovered during order-graph simplification.
#[derive(Clone, Debug)]
pub enum OrderGraphError {
    /// An order node has an empty range of possible values.
    EmptyRange(OrderNode),
    /// A cycle of inequalities exists with non-zero offset. One node
    /// in the cycle is named.
    OrderCycle(OrderNode),
    /// Two or more edges conflict originating from the given node and
    /// going to the given node.
    ConflictingEdges(OrderNode, OrderNode),
    /// Two or more constant ranges conflict on the given node.
    ConstantConflict(OrderNode),
}

/// Result type alias for methods returning order-graph errors.
pub type OrderResult<T> = std::result::Result<T, OrderGraphError>;

/// Either a node ID or a constant value.
#[derive(Clone, Copy, Debug)]
pub enum OrderNodeOrConst {
    /// A symbolic order node.
    Node(OrderNode),
    /// A constant value.
    Const(u64),
}

impl OrderGraph {
    /// Allocate a fresh variable and create a new order node for it.
    pub fn fresh_var(&mut self) -> OrderNode {
        let index = self.next_var;
        self.next_var += 1;
        let var = OrderVar::from_u32(index);
        self.nodes.push(OrderNodeData {
            bound: ConstantBound::None,
            exprs: vec![var],
        })
    }

    /// Mark one node, plus an offset, as equal to the other.
    pub fn eq(&mut self, a: OrderNode, b: OrderNode, offset: u64) {
        self.edges.insert(OrderEdge {
            kind: OrderEdgeKind::Eq,
            lhs: a,
            rhs: b,
            offset,
        });
    }

    /// Mark one node, plus an offset, as less than or equal to the
    /// other.
    pub fn le(&mut self, a: OrderNode, b: OrderNode, offset: u64) {
        self.edges.insert(OrderEdge {
            kind: OrderEdgeKind::Le,
            lhs: a,
            rhs: b,
            offset,
        });
    }

    /// Mark a node as having a fixed constant value.
    pub fn constant(&mut self, a: OrderNode, value: u64) -> OrderResult<()> {
        let new = ConstantBound::merge(self.nodes[a].bound, ConstantBound::Eq(value))
            .ok_or_else(|| OrderGraphError::ConstantConflict(a))?;
        self.nodes[a].bound = new;
        Ok(())
    }

    /// Mark a node as having a fixed upper constant-value bound
    /// (inclusive).
    pub fn le_constant(&mut self, a: OrderNode, hi: u64) -> OrderResult<()> {
        let new = ConstantBound::merge(self.nodes[a].bound, ConstantBound::Le(hi))
            .ok_or_else(|| OrderGraphError::ConstantConflict(a))?;
        self.nodes[a].bound = new;
        Ok(())
    }

    /// Simplify the order graph, potentially finding a contradiction
    /// and returning an error.
    pub fn simplify(&mut self) -> OrderResult<()> {
        const MAX_ITERS: usize = 10;
        
        let mut uf = UnionFind::with_capacity(self.nodes.len());
        for node_idx in self.nodes.keys() {
            uf.add(node_idx);
        }
        let mut merged = vec![];
        let mut iters = 0;
        while self.simplify_iter(&mut uf, &mut merged)? && iters < MAX_ITERS {
            iters += 1;
            merged.sort_unstable();
            merged.dedup();
            for merged_node in merged.drain(..) {
                let merged_into = uf.find_and_update(merged_node);
                if merged_into == merged_node {
                    continue;
                }
                let bound = self.nodes[merged_node].bound;
                let bound = ConstantBound::merge(bound, self.nodes[merged_into].bound)
                    .ok_or_else(|| OrderGraphError::ConstantConflict(merged_into))?;
                self.nodes[merged_into].bound = bound;
                let exprs = std::mem::take(&mut self.nodes[merged_node].exprs);
                self.nodes[merged_into].exprs.extend(exprs.into_iter());
                self.nodes[merged_into].exprs.sort_unstable();
                self.nodes[merged_into].exprs.dedup();
            }
        }
        Ok(())
    }

    /// One fixpoint iteration of node simplification. Returns `true`
    /// if any node was updated or merged.
    fn simplify_iter(
        &mut self,
        uf: &mut UnionFind<OrderNode>,
        merged: &mut Vec<OrderNode>,
    ) -> OrderResult<bool> {
        let mut changed = false;

        // Take all edges, rewrite endpoints according to union-find, and sort.
        let mut edges = Vec::with_capacity(self.edges.len());
        for edge in &self.edges {
            let mut edge = edge.clone();
            let (orig_lhs, orig_rhs) = (edge.lhs, edge.rhs);
            edge.lhs = uf.find_and_update(edge.lhs);
            edge.rhs = uf.find_and_update(edge.rhs);
            changed |= orig_lhs != edge.lhs || orig_rhs != edge.rhs;
            if edge.lhs != edge.rhs {
                edges.push(edge);
            }
        }
        edges.sort();

        // Merge edges where possible.  Iterate through edges grouped
        // by destination node. Note that `src` and `dest` are the
        // first fields in the edge struct, hence iteration order with
        // the derived `Ord` ensures we see edges between the same two
        // nodes sequentially.
        let mut new_edges = BTreeSet::new();
        let mut best_edge: Option<OrderEdge> = None;
        for edge in edges {
            if let Some(e) = best_edge {
                if (edge.lhs, edge.rhs) != (e.lhs, e.rhs) {
                    new_edges.insert(e);
                    best_edge = None;
                }
            }
            match best_edge {
                None => {
                    best_edge = Some(edge);
                }
                Some(cur) => {
                    assert!(cur != edge);
                    changed = true;
                    best_edge =
                        Some(OrderEdge::merge(&cur, &edge).ok_or_else(|| {
                            OrderGraphError::ConflictingEdges(edge.lhs, edge.rhs)
                        })?);
                }
            }
        }
        if let Some(best_edge) = best_edge {
            new_edges.insert(best_edge);
        }
        self.edges = new_edges;

        // Update each node's constant constraint based on edges.
        for edge in &self.edges {
            assert_ne!(edge.lhs, edge.rhs);
            let bound_rhs = OrderEdge::compose_left(&self.nodes[edge.lhs].bound, edge);
            let bound_rhs = ConstantBound::merge(bound_rhs, self.nodes[edge.rhs].bound)
                .ok_or_else(|| OrderGraphError::ConstantConflict(edge.rhs))?;
            changed |= bound_rhs != self.nodes[edge.rhs].bound;
            self.nodes[edge.rhs].bound = bound_rhs;
            let bound_lhs = OrderEdge::compose_right(edge, &self.nodes[edge.rhs].bound);
            let bound_lhs = ConstantBound::merge(bound_lhs, self.nodes[edge.lhs].bound)
                .ok_or_else(|| OrderGraphError::ConstantConflict(edge.lhs))?;
            changed |= bound_lhs != self.nodes[edge.lhs].bound;
            self.nodes[edge.lhs].bound = bound_lhs;
        }

        // If any two nodes have an equality edge with offset 0, merge
        // them.
        for edge in &self.edges {
            if edge.kind == OrderEdgeKind::Eq && edge.offset == 0 {
                changed = true;
                uf.union(edge.lhs, edge.rhs);
                merged.push(edge.lhs);
                merged.push(edge.rhs);
            }
        }

        // If any two nodes have the same constant value, merge them.
        let mut const_values = FxHashMap::default();
        for (node, data) in self.nodes.iter() {
            if let ConstantBound::Eq(x) = data.bound {
                match const_values.entry(x) {
                    Entry::Vacant(v) => {
                        v.insert(node);
                    }
                    Entry::Occupied(o) => {
                        let orig = *o.get();
                        uf.union(orig, node);
                        merged.push(orig);
                        merged.push(node);
                    }
                }
            }
        }

        // TODO: look for inequality cycles with nonzero distance and
        // flag an error (contradiction) if so. Inequality cycles with
        // zero distance merge all the nodes in the cycle.        

        Ok(changed)
    }

    /// Query the order graph, determining whether a given node is
    /// {equal to, less than or equal to} another node or constant
    /// value with at least a certain offset.
    pub fn query(
        &self,
        _lhs: OrderNode,
        _rhs: OrderNodeOrConst,
        _kind: OrderEdgeKind,
        _offset: u64,
    ) -> bool {
        todo!()
    }
}
