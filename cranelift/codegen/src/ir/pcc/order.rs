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
use std::collections::BTreeSet;

/// An Order Graph: represents a shared context in which queries may be
/// made about which values are known to be less-than-or-equal-to other
/// values. Encodes a partial order, with edge labels for minimum
/// distance and for predicates (one edge true only if another
/// comparison is true). Referenced by proof-carrying-code facts.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct OrderGraph {
    /// Next order node index to allocate.
    next_node: u32,

    /// Edges between Order Nodes.
    pub edges: BTreeSet<OrderEdge>,
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
            next_node: 0,
            edges: BTreeSet::new(),
        }
    }

    /// Add a new node to an OrderGraph.
    pub fn add_node(&mut self) -> OrderNode {
        let ret = OrderNode::from_u32(self.next_node);
        self.next_node += 1;
        ret
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
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic_order() {
        let mut o = OrderGraph::new();
        let n1 = o.add_node();
        let n2 = o.add_node();
        let n3 = o.add_node();
        let n4 = o.add_node();
        let n5 = o.add_node();
        let n6 = o.add_node();
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
}
