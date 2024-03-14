//! Order-graph representation of the known equalities and inequalities
//! within a function.
//!
//! In order to state facts about SSA values that can be used to prove
//! boundedness for proof-carrying code (PCC), we need to be able to
//! represent inequalities, and sometimes equalities too.
//!
//! The basic idea is to represent all values that are known be be equal
//! -- that is, equivalence classes -- with one node, and then store
//! edges between nodes when one node is known to be less than or equal
//! to another.
//!
//! Each node can also have a concrete integer value attached to it, if
//! known. Nodes do not contain an explicit representation of the
//! equivalence class's membership set; rather, that set is implicit as
//! the set of all pointers to the node.
//!
//! This encodes an order relation (technically, a non-strict preorder).
//! In particular, there is *transitivity*: if `a <= b` and `b <= c`,
//! then we know `a <= c`. We do not fill out the graph eagerly with the
//! transitive closure of all edges; rather, we expand the graph lazily
//! as we query it, in a sort of analogous way to a union-find tree's
//! "path compression".
//!
//! The edges can be labeled in two ways:
//!
//! 1. An edge always has a "minimum distance" integer label. When zero,
//!    the edge denotes the loose inequality `<=` exactly. When greater
//!    than zero, it specifies a minimum separation; that is, an edge
//!    from `x` to `y` with label `k` means that `x + k <= y`.
//!
//! 2. An edge may have one "predicate" attached, expressed as another
//!    comparison pair. This indicates that the edge's implied
//!    inequality is only true when the given other comparison is true.
//!    Queries may be performed under a specified-true predicate.

use crate::ir::OrderNode;
use cranelift_entity::PrimaryMap;
use std::collections::BTreeSet;

/// An Order Graph: represents a shared context in which queries may be
/// made about which values are known to be less-than-or-equal-to other
/// values. Encodes a partial order, with edge labels for minimum
/// distance and for predicates (one edge true only if another
/// comparison is true). Referenced by proof-carrying-code facts.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct OrderGraph {
    /// Order nodes: arbitrary symbolic values that can be compared.
    pub nodes: PrimaryMap<OrderNode, OrderNodeData>,
    /// Edges between Order Nodes.
    pub edges: BTreeSet<OrderEdge>,
}

/// An "order node": a value in the program about which we know some
/// ordering relations, and for which we may know a specific constant
/// value.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct OrderNodeData {
    /// Known minimum static value.
    pub min: u64,
    /// Known maximum static value.
    pub max: u64,
}

/// An edge from the given node to another node, with the given
/// minimum distance, optionally predicated on some other edge.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct OrderEdge {
    /// The originating node for this ordering-relation edge.
    pub from: OrderNode,
    /// The other node to which this edge's originating node is
    /// less-than-or-equal.
    pub to: OrderNode,
    /// The known minimum value gap. (A pure loose-inequality
    /// corresponds to 0; a strict-inequality corresponds to 1, since we
    /// deal with integers; any higher value specifies a correspondingly
    /// tighter constraint.)
    pub distance: u64,
    /// Another comparison pair that, if true, enables this edge.
    pub pred: Option<(OrderNode, OrderNode)>,
}

impl OrderEdge {
    /// The lowest-sorted OrderEdge for a given from-node, to allow
    /// constructing query ranges in the edge-set.
    fn first_from(from: OrderNode) -> Self {
        OrderEdge {
            from,
            to: OrderNode::from_bits(0),
            distance: 0,
            pred: None,
        }
    }

    /// The highest-sorted OrderEdge for a given from-node, to allow
    /// constructing query ranges in the edge-set.
    fn last_from(from: OrderNode) -> Self {
        OrderEdge {
            from,
            to: OrderNode::from_bits(u32::MAX),
            distance: u64::MAX,
            pred: Some((
                OrderNode::from_bits(u32::MAX),
                OrderNode::from_bits(u32::MAX),
            )),
        }
    }

    /// The lowest-sorted OrderEdge for a given (from, to) pair, to
    /// allow constructing query ranges in the edge-set.
    fn first_from_to(from: OrderNode, to: OrderNode) -> Self {
        OrderEdge {
            from,
            to,
            distance: 0,
            pred: None,
        }
    }

    /// The highest-sorted OrderEdge for a given (from, to) pair, to
    /// allow constructing query ranges in the edge-set.
    fn last_from_to(from: OrderNode, to: OrderNode) -> Self {
        OrderEdge {
            from,
            to,
            distance: u64::MAX,
            pred: Some((
                OrderNode::from_bits(u32::MAX),
                OrderNode::from_bits(u32::MAX),
            )),
        }
    }
}

impl OrderGraph {
    /// Create a new OrderGraph.
    pub fn new() -> Self {
        Self {
            nodes: PrimaryMap::default(),
            edges: BTreeSet::new(),
        }
    }

    /// Add a new symbolic node to an OrderGraph.
    pub fn symbolic(&mut self) -> OrderNode {
        self.nodes.push(OrderNodeData {
            min: 0,
            max: u64::MAX,
        })
    }

    /// Add a new exact node to an OrderGraph.
    pub fn exact(&mut self, value: u64) -> OrderNode {
        self.nodes.push(OrderNodeData {
            min: value,
            max: value,
        })
    }

    /// Add a new node with a static upper bound to an OrderGraph.
    pub fn static_upper_bound(&mut self, max: u64) -> OrderNode {
        self.nodes.push(OrderNodeData { min: 0, max })
    }

    /// Add a new edge between two nodes, optionally with some distance,
    /// optionally predicated on two other nodes being ordered.
    pub fn add_edge(
        &mut self,
        from: OrderNode,
        to: OrderNode,
        distance: u64,
        pred: Option<(OrderNode, OrderNode)>,
    ) {
        self.edges.insert(OrderEdge {
            from,
            to,
            distance,
            pred,
        });
    }

    /// Finalize the order-graph: compute the transitive closure.
    pub fn finalize(&mut self) {
        // Find the transitive closure of inequalities via the
        // "semi-naive evaluation algorithm" (a fixpoint algorithm with
        // batches).
        let mut to_process = self.edges.clone();
        let mut new_edges = BTreeSet::new();
        while !to_process.is_empty() {
            while let Some(edge) = to_process.pop_first() {
                // Find edges to compose with this edge.
                let to_range = self
                    .edges
                    .range(OrderEdge::first_from(edge.to)..=OrderEdge::last_from(edge.to));
                for to_edge in to_range {
                    if let Some(composed) = Self::compose(&edge, to_edge) {
                        if !self.edges.contains(&composed) {
                            new_edges.insert(composed);
                        }
                    }
                }
            }

            to_process = new_edges.clone();
            self.edges.append(&mut new_edges);
        }

        // Propagate min and max across all edges in the transitive
        // closure.
        for edge in &self.edges {
            if edge.pred.is_some() {
                continue;
            }
            let from_max = self.nodes[edge.to]
                .max
                .checked_sub(edge.distance)
                .unwrap_or(0);
            let to_min = self.nodes[edge.from]
                .min
                .checked_add(edge.distance)
                .unwrap_or(u64::MAX);
            self.nodes[edge.from].max = std::cmp::min(self.nodes[edge.from].max, from_max);
            self.nodes[edge.to].min = std::cmp::max(self.nodes[edge.to].min, to_min);
        }
    }

    fn compose(e1: &OrderEdge, e2: &OrderEdge) -> Option<OrderEdge> {
        assert_eq!(e1.to, e2.from);
        if e1.pred.is_some() && e2.pred.is_some() && e1.pred != e2.pred {
            None
        } else {
            Some(OrderEdge {
                from: e1.from,
                to: e2.to,
                distance: e1.distance.checked_add(e2.distance)?,
                pred: e1.pred.or(e2.pred),
            })
        }
    }

    /// Query: can we conclude that the two nodes are ordered with
    /// separation at least the given amount, possibly under the
    /// assumption of another pair being ordered?
    pub fn query(
        &self,
        from: OrderNode,
        to: OrderNode,
        pred: Option<(OrderNode, OrderNode)>,
    ) -> Option<u64> {
        let best_edge = self
            .edges
            .range(OrderEdge::first_from_to(from, to)..=OrderEdge::last_from_to(from, to))
            .filter(|e| e.pred.is_none() || e.pred == pred)
            .next();

        best_edge.map(|e| e.distance)
    }

    /// Get the lower and upper bounds on a node.
    pub fn bounds(&self, node: OrderNode) -> (u64, u64) {
        (self.nodes[node].min, self.nodes[node].max)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sym_order() {
        let mut o = OrderGraph::new();
        let n1 = o.symbolic();
        let n2 = o.symbolic();
        let n3 = o.symbolic();
        let n4 = o.symbolic();
        let n5 = o.symbolic();
        let n6 = o.symbolic();
        o.add_edge(n1, n2, 0, None);
        o.add_edge(n2, n3, 10, Some((n5, n6)));
        o.add_edge(n3, n4, 5, None);
        o.add_edge(n4, n5, 5, Some((n1, n2)));
        o.finalize();
        assert_eq!(o.query(n1, n4, Some((n5, n6))), Some(15));
        assert_eq!(o.query(n1, n4, None), None);
        assert_eq!(o.query(n1, n5, Some((n5, n6))), None);

        o.add_edge(n2, n4, 7, None);
        o.finalize();
        assert_eq!(o.query(n1, n4, None), Some(7));
    }

    #[test]
    fn min_max_values() {
        let mut o = OrderGraph::new();
        let n1 = o.exact(42);
        let n2 = o.symbolic();
        let n3 = o.static_upper_bound(100);
        let n4 = o.static_upper_bound(90);
        o.add_edge(n1, n2, 5, None);
        o.add_edge(n2, n3, 10, None);
        o.add_edge(n3, n4, 0, None);
        o.finalize();
        assert_eq!(o.bounds(n2), (47, 80));
    }
}
