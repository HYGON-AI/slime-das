# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Import guard for a disabled optional Megatron gradient-fusion extension.

HCU integration tests pass ``--no-gradient-accumulation-fusion``.  Some HCU
Megatron releases nevertheless import the optional Apex extension at module
load time.  This module lets that import complete without introducing an
NVIDIA ``libcuda.so.1`` dependency.  Any unexpected attempt to execute the
disabled extension fails immediately with a clear message.
"""


def _fusion_is_disabled(*_args, **_kwargs):
    raise RuntimeError(
        "fused_weight_gradient_mlp_cuda was called even though "
        "--no-gradient-accumulation-fusion is required for this HCU test"
    )


wgrad_gemm_accum_fp32 = _fusion_is_disabled
wgrad_gemm_accum_fp16 = _fusion_is_disabled
