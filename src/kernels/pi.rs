//! FFI to the partition-invariant (PI) CUDA kernels in `src/kernels/pi/`.
//!
//! In deterministic mode every projection goes through [`linear`] and attention
//! through the `pi` backend. Their results are bitwise independent of batch
//! composition, KV split counts, chunked-prefill cut points, prefix-cache state
//! and CUDA-graph padding: each output row is computed by one fixed reduction
//! tree that depends only on the row's own data and positions.

use std::sync::atomic::{AtomicBool, Ordering};

use tch::{Device, Kind, Tensor};

static DETERMINISTIC: AtomicBool = AtomicBool::new(false);

/// Selects the row-invariant GEMM for every [`crate::layers::linear`] call.
pub fn set_deterministic(enabled: bool) {
    DETERMINISTIC.store(enabled, Ordering::Relaxed);
}

pub fn deterministic() -> bool {
    DETERMINISTIC.load(Ordering::Relaxed)
}

pub fn available() -> bool {
    cfg!(has_pi)
}

#[cfg(has_pi)]
mod ffi {
    use std::ffi::{c_char, c_void};
    unsafe extern "C" {
        pub fn pi_current_stream(device: i64) -> *mut c_void;
        pub fn pi_linear(
            x: *const c_void,
            ldx: i64,
            w: *const c_void,
            y: *mut c_void,
            m: i32,
            n: i32,
            k: i32,
            out_fp32: i32,
            stream: *mut c_void,
        ) -> *const c_char;
        pub fn pi_attn_decode_splits(batch: i32, num_kv_heads: i32, max_chunks: i32) -> i32;
        pub fn pi_attn_decode(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            page_table: *const i32,
            table_stride: i64,
            seqlens: *const i32,
            ws_o: *mut f32,
            ws_ml: *mut f32,
            max_chunks: i32,
            out: *mut c_void,
            batch: i32,
            num_qo_heads: i32,
            num_kv_heads: i32,
            page_size: i32,
            splits: i32,
            sm_scale: f64,
            stream: *mut c_void,
        ) -> *const c_char;
        pub fn pi_attn_prefill_plan(
            batch: i32,
            num_qo_heads: i32,
            num_kv_heads: i32,
            max_q_len: i32,
            max_kv_len: i32,
            variant: *mut i32,
            splits: *mut i32,
            max_groups: *mut i32,
        );
        pub fn pi_attn_prefill(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            qo_indptr: *const i32,
            page_table: *const i32,
            table_stride: i64,
            prefix_lens: *const i32,
            out: *mut c_void,
            batch: i32,
            total_q: i32,
            num_qo_heads: i32,
            num_kv_heads: i32,
            page_size: i32,
            max_q_len: i32,
            sm_scale: f64,
            variant: i32,
            splits: i32,
            ws_o: *mut f32,
            ws_ml: *mut f32,
            max_groups: i32,
            stream: *mut c_void,
        ) -> *const c_char;
    }
}

#[cfg(has_pi)]
fn check(message: *const std::ffi::c_char) -> Result<(), String> {
    if message.is_null() {
        Ok(())
    } else {
        Err(unsafe { std::ffi::CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned())
    }
}

#[cfg(has_pi)]
fn stream_of(t: &Tensor) -> Result<*mut std::ffi::c_void, String> {
    match t.device() {
        Device::Cuda(index) => Ok(unsafe { ffi::pi_current_stream(index as i64) }),
        _ => Err("PI kernels require CUDA tensors".into()),
    }
}

fn require(t: &Tensor, kind: Kind, what: &str) -> Result<(), String> {
    if t.kind() != kind || !matches!(t.device(), Device::Cuda(_)) {
        return Err(format!("{what}: expected CUDA {kind:?}, got {:?} on {:?}", t.kind(), t.device()));
    }
    Ok(())
}

/// `x[.., K] @ weight[N, K]^T` with the row-invariant GEMM. Panics on misuse:
/// deterministic mode validates dtypes and shapes when the model is built.
pub(crate) fn linear(x: &Tensor, weight: &Tensor) -> Tensor {
    try_linear(x, weight).unwrap_or_else(|error| panic!("deterministic linear: {error}"))
}

pub(crate) fn try_linear(x: &Tensor, weight: &Tensor) -> Result<Tensor, String> {
    require(x, Kind::BFloat16, "linear input")?;
    require(weight, Kind::BFloat16, "linear weight")?;
    let (n, k) = (weight.size()[0], weight.size()[1]);
    let shape = x.size();
    if shape.last() != Some(&k) || !weight.is_contiguous() {
        return Err(format!("shape mismatch: x {shape:?}, weight {:?}", weight.size()));
    }
    let m: i64 = shape[..shape.len() - 1].iter().product();
    let mut x2 = x.reshape([m, k]);
    if m > 0 && x2.stride()[1] != 1 {
        x2 = x2.contiguous();
    }
    let y = Tensor::empty([m, n], (Kind::BFloat16, weight.device()));
    #[cfg(has_pi)]
    check(unsafe {
        ffi::pi_linear(
            x2.data_ptr(),
            if m > 0 { x2.stride()[0] } else { k },
            weight.data_ptr(),
            y.data_ptr(),
            m as i32,
            n as i32,
            k as i32,
            0,
            stream_of(weight)?,
        )
    })?;
    #[cfg(not(has_pi))]
    return Err("PI kernels were not built".into());
    let mut out_shape = shape;
    *out_shape.last_mut().unwrap() = n;
    Ok(y.reshape(out_shape))
}

/// Chunk capacity of a dense page table: chunks of 64 keys covering every column.
pub(crate) fn max_chunks(table_columns: i64, page_size: i64) -> i64 {
    (table_columns * page_size + 63) / 64
}

pub(crate) fn decode_splits(batch: i64, num_kv_heads: i64, max_chunks: i64) -> i32 {
    #[cfg(has_pi)]
    return unsafe { ffi::pi_attn_decode_splits(batch as i32, num_kv_heads as i32, max_chunks as i32) };
    #[cfg(not(has_pi))]
    {
        let _ = (batch, num_kv_heads, max_chunks);
        1
    }
}

/// Per-chunk decode partials: o `[rows, Hq, max_chunks, 128]`, ml `[rows, Hq, max_chunks, 2]` f32.
pub(crate) struct DecodeWorkspace {
    pub o: Tensor,
    pub ml: Tensor,
}

impl DecodeWorkspace {
    pub fn new(rows: i64, num_heads: i64, max_chunks: i64, device: Device) -> Self {
        Self {
            o: Tensor::empty([rows, num_heads, max_chunks, 128], (Kind::Float, device)),
            ml: Tensor::empty([rows, num_heads, max_chunks, 2], (Kind::Float, device)),
        }
    }
    pub fn rows(&self) -> i64 {
        self.o.size()[0]
    }
    pub fn max_chunks(&self) -> i64 {
        self.o.size()[2]
    }
}

/// Decode attention for `q [batch, Hq, 128]` against the paged caches; writes `out [batch, Hq*128]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    page_table: &Tensor,
    seqlens: &Tensor,
    workspace: &DecodeWorkspace,
    splits: i32,
    num_kv_heads: i64,
    out: &Tensor,
) -> Result<(), String> {
    let (batch, num_heads) = (q.size()[0], q.size()[1]);
    require(page_table, Kind::Int, "page table")?;
    require(seqlens, Kind::Int, "sequence lengths")?;
    if page_table.size()[0] < batch || page_table.stride()[1] != 1 || workspace.rows() < batch {
        return Err("decode page table or workspace smaller than the batch".into());
    }
    let page_size = k_cache.size()[1];
    if workspace.max_chunks() < max_chunks(page_table.size()[1], page_size) {
        return Err("decode workspace has fewer chunks than the page table".into());
    }
    #[cfg(has_pi)]
    return check(unsafe {
        ffi::pi_attn_decode(
            q.data_ptr(),
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            page_table.data_ptr().cast(),
            page_table.stride()[0],
            seqlens.data_ptr().cast(),
            workspace.o.data_ptr().cast(),
            workspace.ml.data_ptr().cast(),
            workspace.max_chunks() as i32,
            out.data_ptr(),
            batch as i32,
            num_heads as i32,
            num_kv_heads as i32,
            page_size as i32,
            splits,
            1.0 / (q.size()[2] as f64).sqrt(),
            stream_of(q)?,
        )
    });
    #[cfg(not(has_pi))]
    {
        let _ = (k_cache, v_cache, seqlens, workspace, splits, num_kv_heads, out, num_heads);
        Err("PI kernels were not built".into())
    }
}

/// Host-side prefill plan; all fields come from host metadata (no GPU reads).
pub(crate) struct PrefillPlan {
    pub variant: i32,
    pub splits: i32,
    pub max_groups: i32,
    pub max_q_len: i32,
    pub total_q: i32,
    pub workspace: Option<(Tensor, Tensor)>,
}

pub(crate) fn plan_prefill(
    q_lens: &[i64],
    prefixes: &[i64],
    num_heads: i64,
    num_kv_heads: i64,
    device: Device,
) -> PrefillPlan {
    let max_q_len = q_lens.iter().copied().max().unwrap_or(0).max(1) as i32;
    let max_kv_len = q_lens
        .iter()
        .zip(prefixes)
        .map(|(q, p)| q + p)
        .max()
        .unwrap_or(1)
        .max(1) as i32;
    let total_q: i64 = q_lens.iter().sum();
    let (mut variant, mut splits, mut max_groups) = (2, 1, 1);
    #[cfg(has_pi)]
    unsafe {
        ffi::pi_attn_prefill_plan(
            q_lens.len() as i32,
            num_heads as i32,
            num_kv_heads as i32,
            max_q_len,
            max_kv_len,
            &mut variant,
            &mut splits,
            &mut max_groups,
        )
    };
    let workspace = (splits > 1).then(|| {
        (
            Tensor::empty([total_q, num_heads, max_groups as i64, 128], (Kind::Float, device)),
            Tensor::empty([total_q, num_heads, max_groups as i64, 2], (Kind::Float, device)),
        )
    });
    PrefillPlan {
        variant,
        splits,
        max_groups,
        max_q_len,
        total_q: total_q as i32,
        workspace,
    }
}

/// Prefill/extend attention for `q [total_q, Hq, 128]`; K/V of every token are
/// already in the paged caches. Writes `out` (`[total_q, Hq*128]`, contiguous).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    qo_indptr: &Tensor,
    page_table: &Tensor,
    prefix_lens: &Tensor,
    plan: &PrefillPlan,
    num_kv_heads: i64,
    out: &Tensor,
) -> Result<(), String> {
    require(qo_indptr, Kind::Int, "cu_seqlens_q")?;
    require(prefix_lens, Kind::Int, "prefix_lens")?;
    require(page_table, Kind::Int, "page table")?;
    if page_table.stride()[1] != 1 || q.size()[0] != plan.total_q as i64 {
        return Err("prefill metadata does not match the query".into());
    }
    let (ws_o, ws_ml) = plan
        .workspace
        .as_ref()
        .map_or((std::ptr::null_mut(), std::ptr::null_mut()), |(o, ml)| {
            (o.data_ptr().cast::<f32>(), ml.data_ptr().cast::<f32>())
        });
    #[cfg(has_pi)]
    return check(unsafe {
        ffi::pi_attn_prefill(
            q.data_ptr(),
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            qo_indptr.data_ptr().cast(),
            page_table.data_ptr().cast(),
            page_table.stride()[0],
            prefix_lens.data_ptr().cast(),
            out.data_ptr(),
            prefix_lens.size()[0] as i32,
            plan.total_q,
            q.size()[1] as i32,
            num_kv_heads as i32,
            k_cache.size()[1] as i32,
            plan.max_q_len,
            1.0 / (q.size()[2] as f64).sqrt(),
            plan.variant,
            plan.splits,
            ws_o,
            ws_ml,
            plan.max_groups,
            stream_of(q)?,
        )
    });
    #[cfg(not(has_pi))]
    {
        let _ = (k_cache, v_cache, num_kv_heads, out, ws_o, ws_ml);
        Err("PI kernels were not built".into())
    }
}
