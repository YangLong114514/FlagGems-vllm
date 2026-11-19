# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""KDA RecoverSSM verify: native/Gems pairs timed with device events.

Both kernels write ``out`` and the verify records deterministically on every
call and never mutate the checkpoint pool, so no per-sample state restore is
needed. SpeedUp = latency_native / latency_gems (repo convention).
"""

import math
import statistics

import pytest
import torch

import flaggems_vllm as gems
from baseline import kda_recoverssm_verify as baseline
from benchmark.base import Benchmark
from benchmark.conftest import Config
from benchmark.consts import BenchMode
from flaggems_vllm.runtime import torch_device_fn
from tests.test_kda_recoverssm_verify import (
    OUT_TOL,
    assert_records_match,
    reference,
)


def _measure(fn, data):
    for _ in range(5):
        fn(**data)
    torch_device_fn.synchronize()
    samples = []
    for _ in range(25):
        torch_device_fn.synchronize()
        start = torch_device_fn.Event(enable_timing=True)
        end = torch_device_fn.Event(enable_timing=True)
        start.record()
        fn(**data)
        end.record()
        torch_device_fn.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _native_slot(**data):
    raise RuntimeError("native slot must be resolved by get_latency")


class KDARecoverSSMVerifyBenchmark(Benchmark):
    DEFAULT_DTYPES = baseline.DTYPES
    DEFAULT_SHAPES = [(case.name,) for case in baseline.PERF_CASES]
    DEFAULT_SHAPE_DESC = "case (baseline.PERF_CASES)"

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def init_user_config(self):
        if Config.mode != BenchMode.KERNEL:
            raise ValueError(
                "RecoverSSM verify uses --mode kernel with no-Graph device events"
            )
        super().init_user_config()

    def run(self):
        self.native_op = baseline.native()
        super().run()

    def get_input_iter(self, dtype):
        cases = {case.name: case for case in baseline.PERF_CASES}
        for (name,) in self.shapes:
            self.case = cases[name]
            self.dtype = dtype
            self.latencies = None
            yield (baseline.build(self.case, dtype, self.device),)

    def record_shapes(self, *args, **kwargs):
        return dict(case=self.case.name, inputs=super().record_shapes(*args, **kwargs))

    def get_latency(self, op, *args, **kwargs):
        if self.latencies is None:
            self.latencies = self._measure_pair(kwargs)
        return self.latencies["native_ms" if op is self.torch_op else "gems_ms"]

    def _measure_pair(self, data):
        expected, ref_corr, written = reference(self.case, data)
        row = dict(
            case=self.case.name,
            dtype=str(self.dtype),
            backend=gems.vendor_name,
            graph=False,
            unit="ms",
            protocol="device-event-5warmup-25sample-median",
            config=baseline.config({"case": self.case}, self.dtype),
            candidate_accuracy="NOT_RUN",
            native_accuracy="NOT_RUN",
            native_ms=None,
            selected_source=None,
        )
        functions = {"gems": self.gems_op}
        if "latency_base" in self.to_bench_metrics:
            functions["native"] = self.native_op

        candidate_error = None
        for name, fn in functions.items():
            key = "candidate_accuracy" if name == "gems" else "native_accuracy"
            try:
                out = fn(**data)
                torch_device_fn.synchronize()
                torch.testing.assert_close(
                    out.to(torch.float64),
                    expected,
                    rtol=OUT_TOL[self.dtype],
                    atol=OUT_TOL[self.dtype],
                )
                assert_records_match(
                    self.case, data, data["records"], ref_corr, written
                )
                row[key] = "PASS"
                samples = _measure(fn, data)
                latency = statistics.median(samples)
                if not math.isfinite(latency) or latency <= 0:
                    raise ValueError(f"invalid {name} latency: {latency}")
                row[name + "_samples_ms"] = samples
                row[name + "_ms"] = latency
            except Exception as error:
                if row[key] != "PASS":
                    row[key] = "FAIL"
                    row[name + "_error"] = repr(error)
                else:
                    row[name + "_timing_error"] = repr(error)
                if name == "gems":
                    candidate_error = error
                else:
                    raise

        if row["native_ms"] is not None and "gems_ms" in row:
            row["native_speedup"] = row["native_ms"] / row["gems_ms"]
            row["selected_source"] = "native"
        if candidate_error is not None:
            raise candidate_error
        return row


@pytest.mark.kda_recoverssm_verify
def test_kda_recoverssm_verify():
    KDARecoverSSMVerifyBenchmark(
        op_name=baseline.OP_NAME,
        torch_op=_native_slot,
        gems_op=gems.kda_recoverssm_verify,
        is_inplace=baseline.IS_INPLACE,
    ).run()
