# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""KDA RecoverSSM verify benchmark using the repository's latency protocols."""

import pytest
import torch

import flaggems_vllm as gems
from baseline import kda_recoverssm_verify as baseline
from benchmark import base
from benchmark import conftest as bench_config
from benchmark.base import Benchmark
from flaggems_vllm.runtime import torch_device_fn
from tests.test_kda_recoverssm_verify import (
    OUT_TOL,
    assert_records_match,
    kda_worker,  # noqa: F401
    records_by_request,
    reference,
)


class KDARecoverSSMVerifyBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPES = [(case.name,) for case in baseline.PRODUCTION_CASES]
    DEFAULT_SHAPE_DESC = "case (baseline.PRODUCTION_CASES)"

    def run(self):
        data = base.Config.base_data
        if not data or self.op_name not in data:
            return super().run()

        # The shared lookup accepts variants and indexes each by first tensor shape.
        # Select the exact collector workload before matching Q and dtype: different
        # batch/window combinations can have the same flattened Q shape.
        rows = data[self.op_name].get("shapes", {})
        variants = {}
        for case in baseline.PERF_CASES:
            key = baseline.key_shape({"case": case})
            if key in rows:
                q_shape = [1, sum(case.lengths), case.heads, case.head_dim]
                variants[key] = {"shapes": {str(q_shape): rows[key]}}

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                base.Config, "base_data", {**data, self.op_name: {"variants": variants}}
            )
            return super().run()

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        cases = {case.name: case for case in baseline.PERF_CASES}
        for (name,) in self.shapes:
            self.case = cases[name]
            self.base_variant = baseline.key_shape({"case": self.case})
            self.dtype = dtype
            self.reference_data = None
            inputs = baseline.build(
                self.case, dtype, self.device, vendor=gems.vendor_name
            )
            yield (inputs,)

    def record_shapes(self, *args, **kwargs):
        return dict(
            case=self.case.name,
            baseline_shape_key=self.base_variant,
            inputs=super().record_shapes(*args, **kwargs),
        )

    def get_latency(self, op, *args, **kwargs):
        if self.reference_data is None:
            self.reference_data = reference(self.case, kwargs)
        expected, ref_corr, written = self.reference_data
        if op is self.torch_op:
            kwargs = baseline.native_inputs(kwargs)
        kwargs["out"] = torch.empty_like(kwargs["v"])
        out = op(**kwargs)
        torch_device_fn.synchronize()
        torch.testing.assert_close(
            out.to(expected.device).to(expected.dtype),
            expected,
            rtol=OUT_TOL[self.dtype],
            atol=OUT_TOL[self.dtype],
        )
        assert_records_match(
            self.case, kwargs, records_by_request(kwargs), ref_corr, written
        )
        return super().get_latency(op, *args, **kwargs)


def _run_benchmark():
    KDARecoverSSMVerifyBenchmark(
        op_name=baseline.OP_NAME,
        torch_op=baseline.native(),
        gems_op=gems.kda_recoverssm_verify,
        is_inplace=baseline.IS_INPLACE,
    ).run()


def _run_metax_benchmark(config):
    # The child has no pytest session. Keep the usual Benchmark.run protocol
    # and forward its report entries to the parent session for recording.
    results, messages = [], []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "Config", config)
        patch.setattr(
            base, "update_result", lambda name, data: results.append((name, data))
        )
        patch.setattr(base, "emit_record_logger", messages.append)
        _run_benchmark()
    return results, messages


@pytest.mark.kda_recoverssm_verify
def test_kda_recoverssm_verify(kda_worker):  # noqa: F811
    if kda_worker is None:
        _run_benchmark()
        return
    results, messages = kda_worker.submit(
        _run_metax_benchmark, bench_config.Config
    ).result()
    for name, data in results:
        bench_config.update_result(name, data)
    for message in messages:
        bench_config.emit_record_logger(message)
