# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""KDA RecoverSSM verify benchmark using the repository's latency protocols."""

import pytest
import torch

import flaggems_vllm as gems
from baseline import kda_recoverssm_verify as baseline
from benchmark.base import Benchmark
from flaggems_vllm.runtime import torch_device_fn
from tests.test_kda_recoverssm_verify import (
    OUT_TOL,
    assert_records_match,
    reference,
)


class KDARecoverSSMVerifyBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPES = [(case.name,) for case in baseline.PRODUCTION_CASES]
    DEFAULT_SHAPE_DESC = "case (baseline.PRODUCTION_CASES)"

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        cases = {case.name: case for case in baseline.PERF_CASES}
        for (name,) in self.shapes:
            self.case = cases[name]
            self.dtype = dtype
            self.reference_data = None
            yield (baseline.build(self.case, dtype, self.device),)

    def record_shapes(self, *args, **kwargs):
        return dict(case=self.case.name, inputs=super().record_shapes(*args, **kwargs))

    def get_latency(self, op, *args, **kwargs):
        if self.reference_data is None:
            self.reference_data = reference(self.case, kwargs)
        expected, ref_corr, written = self.reference_data
        out = op(**kwargs)
        torch_device_fn.synchronize()
        torch.testing.assert_close(
            out.to(torch.float64),
            expected,
            rtol=OUT_TOL[self.dtype],
            atol=OUT_TOL[self.dtype],
        )
        assert_records_match(self.case, kwargs, kwargs["records"], ref_corr, written)
        return super().get_latency(op, *args, **kwargs)


@pytest.mark.kda_recoverssm_verify
def test_kda_recoverssm_verify():
    KDARecoverSSMVerifyBenchmark(
        op_name=baseline.OP_NAME,
        torch_op=baseline.native(),
        gems_op=gems.kda_recoverssm_verify,
        is_inplace=baseline.IS_INPLACE,
    ).run()
