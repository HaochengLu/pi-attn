# Stale results

Kept for history; superseded by results/ (every number in the paper comes from one
eval/run_all.sh regeneration). Do not cite these.

- accuracy/: GSM8K summary of 2026-10-20 (before the fe3d664 regeneration).
- engine_profile/: nsys kernel-time profiles (eval/profile_engine.sh) taken before fe3d664;
  they predate the fused decode fold (still show decode_merge_kernel) and stream-K.
- kernels/decode_variants.*: decode-kernel variant sweep of 2026-10-20 that chose the
  shipped fused decode.
