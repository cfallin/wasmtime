use crate::from_success_code;
use std::io::{Result, Error, ErrorKind};
use std::mem::MaybeUninit;

#[derive(Debug, Copy, Clone)]
pub enum ClockId {
    Realtime,
    Monotonic,
    ProcessCPUTime,
    ThreadCPUTime,
}

impl ClockId {
    #[cfg(not(target_os = "openbsd"))]
    pub fn as_raw(&self) -> Result<libc::clockid_t> {
        match self {
            Self::Realtime => Ok(libc::CLOCK_REALTIME),
            Self::Monotonic => Ok(libc::CLOCK_MONOTONIC),
            Self::ProcessCPUTime => Ok(libc::CLOCK_PROCESS_CPUTIME_ID),
            Self::ThreadCPUTime => Ok(libc::CLOCK_THREAD_CPUTIME_ID),
        }
    }

    #[cfg(target_os = "openbsd")]
    pub fn as_raw(&self) -> Result<libc::clockid_t> {
        match self {
            Self::Realtime => Ok(libc::CLOCK_REALTIME),
            Self::Monotonic => Ok(libc::CLOCK_MONOTONIC),
            Self::ProcessCPUTime => Err(Error::from(ErrorKind::InvalidInput)),
            Self::ThreadCPUTime => Err(Error::from(ErrorKind::InvalidInput)),
        }
    }
}

pub fn clock_getres(clock_id: ClockId) -> Result<libc::timespec> {
    let id = clock_id.as_raw()?;
    let mut timespec = MaybeUninit::<libc::timespec>::uninit();
    from_success_code(unsafe { libc::clock_getres(id, timespec.as_mut_ptr()) })?;
    Ok(unsafe { timespec.assume_init() })
}

pub fn clock_gettime(clock_id: ClockId) -> Result<libc::timespec> {
    let id = clock_id.as_raw()?;
    let mut timespec = MaybeUninit::<libc::timespec>::uninit();
    from_success_code(unsafe { libc::clock_gettime(id, timespec.as_mut_ptr()) })?;
    Ok(unsafe { timespec.assume_init() })
}
