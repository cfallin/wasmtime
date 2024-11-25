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

use crate::entity::{entity_impl, PrimaryMap};
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

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
    /// Next fresh variable to allocate.
    next_var: u32,
}

/// Data at one node in the order graph.
#[derive(Clone, Debug)]
pub struct OrderNodeData {
    /// Known lower-bound constant value, if any.
    pub lo: Option<u64>,
    /// Known upper-bound constant value, if any.
    pub hi: Option<u64>,
    /// Variables in the equivalence class represented by this node.
    pub exprs: Vec<OrderVar>,
    /// Ordering edges to other nodes.
    pub edges: BTreeSet<OrderEdge>,
}

/// An ordering edge from one node to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrderEdge {
    /// The other (right-hand side) node in the ordering relation.
    pub dest: OrderNode,
    /// Kind of edge: equality or inequality.
    pub kind: OrderEdgeKind,
    /// The offset (distance) denoted by this edge.
    pub offset: u64,
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
    /// Allocate a fresh variable.
    pub fn fresh_var(&mut self) -> OrderVar {
        let index = self.next_var;
        self.next_var += 1;
        OrderVar::from_u32(index)
    }

    /// Mark one node, plus an offset, as equal to the other.
    pub fn eq(&mut self, a: OrderNode, b: OrderNode, offset: u64) {
        self.nodes[a].edges.insert(OrderEdge {
            kind: OrderEdgeKind::Eq,
            dest: b,
            offset,
        });
    }

    /// Mark one node, plus an offset, as less than or equal to the
    /// other.
    pub fn le(&mut self, a: OrderNode, b: OrderNode, offset: u64) {
        self.nodes[a].edges.insert(OrderEdge {
            kind: OrderEdgeKind::Le,
            dest: b,
            offset,
        });
    }

    /// Mark a node as having a fixed constant value.
    pub fn constant(&mut self, a: OrderNode, value: u64) -> OrderResult<()> {
        let node = &mut self.nodes[a];
        if node.lo.is_some() && Some(value) < node.lo {
            return Err(OrderGraphError::ConstantConflict(a));
        }
        if node.hi.is_some() && Some(value) > node.hi {
            return Err(OrderGraphError::ConstantConflict(a));
        }
        node.lo = Some(value);
        node.hi = Some(value);
        Ok(())
    }

    /// Mark a node as having a fixed lower or upper (or both)
    /// constant-value bound. Both ends are inclusive.
    pub fn bounded(&mut self, a: OrderNode, lo: Option<u64>, hi: Option<u64>) -> OrderResult<()> {
        let node = &mut self.nodes[a];
        let lo = match (node.lo, lo) {
            (None, lo) | (lo, None) => lo,
            (Some(existing), Some(new)) => Some(std::cmp::max(existing, new)),
        };
        node.lo = lo;
        let hi = match (node.hi, hi) {
            (None, hi) | (hi, None) => hi,
            (Some(existing), Some(new)) => Some(std::cmp::min(existing, new)),
        };
        node.hi = hi;

        if let (Some(lo), Some(hi)) = (node.lo, node.hi) {
            if lo > hi {
                return Err(OrderGraphError::ConstantConflict(a));
            }
        }
        Ok(())
    }

    /// Simplify the order graph, potentially finding a contradiction
    /// and returning an error.
    pub fn simplify(&mut self) -> OrderResult<()> {
        while self.simplify_iter()? {}
        Ok(())
    }

    /// One fixpoint iteration of node simplification. Returns `true`
    /// if any node was updated or merged.
    fn simplify_iter(&mut self) -> OrderResult<bool> {
        let mut changed = false;

        // For each node, merge edges where possible.
        for node in self.nodes.keys() {
            // Iterate through edges grouped by destination node. Note
            // that `dest` is the first field in the edge struct,
            // hence iteration order with the derived `Ord` ensures we
            // see edges to the same dest node sequentially.
            let mut edges = std::mem::take(&mut self.nodes[node].edges);
            let mut new_edges = BTreeSet::new();
            let mut best_edge: Option<OrderEdge> = None;
            for edge in &edges {
                if best_edge.is_some() && edge.dest != best_edge.unwrap().dest {
                    new_edges.insert(best_edge.take().unwrap());
                }
                match best_edge {
                    None => {
                        best_edge = Some(*edge);
                    }
                    Some(cur) => {
                        assert!(cur != *edge);
                        match (cur.kind, edge.kind) {
                            (OrderEdgeKind::Eq, OrderEdgeKind::Eq) => {
                                return Err(OrderGraphError::ConflictingEdges(node, edge.dest));
                            }
                            (OrderEdgeKind::Eq, OrderEdgeKind::Le) if cur.offset < edge.offset => {
                                return Err(OrderGraphError::ConflictingEdges(node, edge.dest));
                            }
                            (OrderEdgeKind::Le, OrderEdgeKind::Eq) if edge.offset < cur.offset => {
                                return Err(OrderGraphError::ConflictingEdges(node, edge.dest));
                            }
                            (OrderEdgeKind::Eq, OrderEdgeKind::Le) => {
                                best_edge = Some(cur);
                            }
                            (OrderEdgeKind::Le, OrderEdgeKind::Eq) => {
                                best_edge = Some(*edge);
                            }
                            (OrderEdgeKind::Le, OrderEdgeKind::Le) => {
                                best_edge = Some(OrderEdge {
                                    kind: OrderEdgeKind::Le,
                                    dest: cur.dest,
                                    offset: std::cmp::max(cur.offset, edge.offset),
                                });
                            }
                        }
                    }
                }
            }
            if let Some(best_edge) = best_edge {
                new_edges.insert(best_edge);
            }
            if new_edges != edges {
                changed = true;
            }
            self.nodes[node].edges = new_edges;
        }

        // Update each node's lo/hi range based on edges.
        let lo_updates: Vec<(OrderNode, u64)> = vec![];
        for node in self.nodes.keys() {
            let mut hi = self.nodes[node].hi;
            // Invariant: we'll have at most one edge per dest node,
            // so deferring dest nodes' `lo` updates is safe (we won't
            // have two conflicting updates).
            for edge in &self.nodes[node].edges {
                if let Some(offset_hi) = hi.map(|val| val.saturating_add(edge.offset)) {
                    let dest_lo = self.nodes[edge.dest].lo.unwrap_or(offset_hi);
                    todo!()
                }
            }

            if hi != self.nodes[node].hi || lo_updates.len() > 0 {
                changed = true;
            }
            self.nodes[node].hi = hi;
            for (node, lo) in lo_updates.drain(..) {
                self.nodes[node].lo = Some(lo);
            }
        }

        // If any two nodes have the same constant value, merge them.

        // For each node, if outgoing edges imply two destination
        // nodes are exactly equal, merge them.

        // Look for inequality cycles with nonzero distance and flag
        // an error (contradiction) if so.

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
