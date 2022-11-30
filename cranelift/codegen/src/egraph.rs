//! Egraph-based mid-end optimization framework.

pub(crate) mod domtree;
pub(crate) mod elaborate;
mod stores;
mod node;


// TODO: use Stats below.
// TODO: use `stores` and do store-to-load forwarding.

#[derive(Clone, Debug, Default)]
pub(crate) struct Stats {
    pub(crate) node_created: u64,
    pub(crate) node_param: u64,
    pub(crate) node_result: u64,
    pub(crate) node_pure: u64,
    pub(crate) node_inst: u64,
    pub(crate) node_load: u64,
    pub(crate) node_dedup_query: u64,
    pub(crate) node_dedup_hit: u64,
    pub(crate) node_dedup_miss: u64,
    pub(crate) node_ctor_created: u64,
    pub(crate) node_ctor_deduped: u64,
    pub(crate) node_union: u64,
    pub(crate) node_subsume: u64,
    pub(crate) store_map_insert: u64,
    pub(crate) side_effect_nodes: u64,
    pub(crate) rewrite_rule_invoked: u64,
    pub(crate) rewrite_depth_limit: u64,
    pub(crate) store_to_load_forward: u64,
    pub(crate) elaborate_visit_node: u64,
    pub(crate) elaborate_memoize_hit: u64,
    pub(crate) elaborate_memoize_miss: u64,
    pub(crate) elaborate_memoize_miss_remat: u64,
    pub(crate) elaborate_licm_hoist: u64,
    pub(crate) elaborate_func: u64,
    pub(crate) elaborate_func_pre_insts: u64,
    pub(crate) elaborate_func_post_insts: u64,
}
