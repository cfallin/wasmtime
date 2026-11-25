//! Access to the store of what is being debugged.

#[cfg(feature = "rr")]
use wasmtime::rr::Replayer;
use wasmtime::{
    AsContext, AsContextMut, BreakpointState, FrameHandle, Func, Instance, Result, Store,
    StoreContext, StoreContextMut,
};

/// How to view a replay's store, which has no host data, as a debuggee's: this
/// is [`Replayer::store`], so there are only replays where `T` is `()`.
#[cfg(feature = "rr")]
pub(crate) type ReplayView<T> = for<'a> fn(&'a Replayer) -> StoreContext<'a, T>;

/// What a debuggee runs.
pub(crate) enum Target<T: 'static> {
    Live(Store<T>),
    #[cfg(feature = "rr")]
    Replay(Replayer, ReplayView<T>),
}

impl<T: Send + 'static> Target<T> {
    pub(crate) fn store(&mut self) -> DebuggeeStore<'_, T> {
        match self {
            Target::Live(store) => DebuggeeStore::live(store.as_context_mut()),
            #[cfg(feature = "rr")]
            Target::Replay(replayer, view) => DebuggeeStore(Access::Replay(replayer, *view)),
        }
    }
}

/// A paused debuggee's store, as given to
/// [`Debuggee::with_store`](crate::Debuggee::with_store).
///
/// A live debuggee's store may be changed, through
/// [`DebuggeeStore::store_mut`]. A replay's store can only be inspected, through
/// [`AsContext`], other than its debugging state, which
/// [`DebuggeeStore::edit_breakpoints`] changes.
pub struct DebuggeeStore<'a, T: 'static>(Access<'a, T>);

enum Access<'a, T: 'static> {
    Live(StoreContextMut<'a, T>),
    #[cfg(feature = "rr")]
    Replay(&'a mut Replayer, ReplayView<T>),
}

impl<'a, T: 'static> DebuggeeStore<'a, T> {
    pub(crate) fn live(store: StoreContextMut<'a, T>) -> Self {
        DebuggeeStore(Access::Live(store))
    }

    pub(crate) fn reborrow(&mut self) -> DebuggeeStore<'_, T> {
        DebuggeeStore(match &mut self.0 {
            Access::Live(store) => Access::Live(store.as_context_mut()),
            #[cfg(feature = "rr")]
            Access::Replay(replayer, view) => Access::Replay(replayer, *view),
        })
    }

    /// Mutable access to the store, or `None` for a replay.
    pub fn store_mut(&mut self) -> Option<StoreContextMut<'_, T>> {
        match &mut self.0 {
            Access::Live(store) => Some(store.as_context_mut()),
            #[cfg(feature = "rr")]
            Access::Replay(..) => None,
        }
    }

    /// The frames of the paused execution; see [`Store::debug_exit_frames`].
    pub fn debug_exit_frames(&mut self) -> Vec<FrameHandle> {
        match &mut self.0 {
            Access::Live(store) => store.debug_exit_frames().collect(),
            #[cfg(feature = "rr")]
            Access::Replay(replayer, _) => replayer.debug_exit_frames(),
        }
    }

    /// A function in `instance`'s function index space; see
    /// [`Instance::debug_function`].
    pub fn debug_function(&mut self, instance: Instance, index: u32) -> Option<Func> {
        match &mut self.0 {
            Access::Live(store) => instance.debug_function(store, index),
            #[cfg(feature = "rr")]
            Access::Replay(replayer, _) => replayer.debug_function(instance, index),
        }
    }

    /// Edits breakpoints, single-stepping, and watchpoints; see
    /// [`Store::edit_breakpoints`].
    pub fn edit_breakpoints<R>(
        &mut self,
        f: impl FnOnce(&mut BreakpointState<'_>) -> Result<R>,
    ) -> Result<R> {
        match &mut self.0 {
            Access::Live(store) => store.as_context_mut().edit_breakpoints(f),
            #[cfg(feature = "rr")]
            Access::Replay(replayer, _) => replayer.edit_breakpoints(f),
        }
    }
}

#[cfg(feature = "rr")]
impl<'a> DebuggeeStore<'a, ()> {
    pub(crate) fn replay(replayer: &'a mut Replayer) -> Self {
        DebuggeeStore(Access::Replay(replayer, Replayer::store))
    }
}

impl<T: 'static> AsContext for DebuggeeStore<'_, T> {
    type Data = T;

    fn as_context(&self) -> StoreContext<'_, T> {
        match &self.0 {
            Access::Live(store) => store.as_context(),
            #[cfg(feature = "rr")]
            Access::Replay(replayer, view) => view(replayer),
        }
    }
}
