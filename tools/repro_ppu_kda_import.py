# SPDX-License-Identifier: Apache-2.0
"""Reproduce the vLLM-SAIL KDA import failure on a PPU container without NVML.

Run in a fresh process: python tools/repro_ppu_kda_import.py
Affected builds raise NVMLError_LibraryNotFound during the import.
"""

if __name__ == "__main__":
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import kda_recoverssm_verify

    print(f"Import succeeded: {kda_recoverssm_verify.__module__}")
