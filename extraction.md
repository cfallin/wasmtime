# Better aegraph extraction

*Problem*: we wish to perform a *multi-extraction*; that is, given an
aegraph, pick one tree (expression) for each of several roots.

*Current solution*: greedy DP approach that, at every union-node,
picks the subtree with lower cost.

  - Issue #1: does not account for structure sharing (DAG rather than
    tree) -- a subtree's cost should be amortized over many uses.
  - Issue #2: does not account for already-committed results -- once
    we start elaborating and codegen the ops for some way of computing
    of a value, that value is "free" and we shouldn't account for its
    cost anymore. (And a small computation on top of that
    already-computed value should be preferred over a whole uncomputed
    tree, even if the latter is locally cheaper.)
    
1. Original proposed solution (the "Wadern approach"):

   Perform extractions interleaved with elaborations. At each root,
   before we elaborate the root, decide how we are going to compute
   each eclass (i.e., pick directions for each union-node).

   Keep a "seen-set" as we traverse. This set is local to the
   traversal over just this root. When we visit a node we've already
   seen, count zero cost. This accounts for structure sharing.
   
   When visiting a node that has been elaborated already, count as
   zero-cost.
   
   Don't do memoization (the DP approach); costs may change as we
   commit or depending on what we've traversed before.
   
2. Attempt at alightly improved "node state" variant that avoids
   allocating a seen-set (the "Iqaluit algorithm"):

   ```rust
   enum NodeState {
     Unused,
     SeenLeaf { generation: u32 },
     SeenUnion { generation: u32, best_child: Value, }
     Committed,
   }
   ```

   We interleave `pick_best` and `commit` when traversing the
   skeleton. `pick_best` sets up `Seen` nodes; `commit` flips to
   `Committed`, using the traced-out path. We increment `generation`
   in the root traversal and treat `Seen` nodes with stale generations
   as `Unused`.
   
   This *works* but doesn't do what we intuitively want -- key
   observation: the backtracking is actually N-level. When visiting
   one side of a Union, we haven't actually committed to having that
   subtree in the seen-set yet; we need to reomve it when we visit the
   other side.
   
   On the other hand, when visiting args of an op, we need to commit
   what the first arg implies when visiting the second arg -- but only
   within the scope of that op. The op itself may be one arm of a
   union and we need to remove its effects from `seen` when visiting
   the other arm.
   
3. The key insight of disjnctions and conjunctions (the "Manitoba
   approach"):
   
   Think of extracting an expression that has conjunctions and
   disjunctions.
   - At a conjunction, we need to choose a best option for each
     argument; we are extracting *all* of them; we account for
     structure sharing from one to the next. Greedy algo picks best
     for each in turn, having committed (within this scope) to the
     priors already.
  - At a disjunction, we need to choose only one of the arguments as
    best; each does not see the effects of the others on `seen`.
    
  We can avoid the two-level notion of "seen" and "committed" by
  seeing the *whole skeleton* as one conjunction. That is, there is
  one root, the program; its arguments are all skeleton ops.
  
  We then need a nested seen-set that supports
  
  - insert(value)
  - snapshot() -> Snapshot
  - restore_to(Snapshot) -> RemovedValues
  - insert_values(RemovedValues)
  
  We can build this with a `Vec` and a `HashSet`; push in insertion
  order into the `Vec`; snapshot is index in vec; pop off back and
  remove from hashset to restore.
  
  Alexa's insight: Optimize for common case where right (second) side
  of disjunction is taken -- we don't need to restore and re-insert
  LHS seen.
  
  TODO: write a functional version of this first to make it clear how
  we're passing versions of `seen` around.
