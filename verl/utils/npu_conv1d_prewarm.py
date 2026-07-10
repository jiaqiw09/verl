# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
_CONV1D_PREWARM_DONE = False
_SEQ_LEN = 10300
_DIM = 3072
_WIDTH = 4
_QUERY_START_LOC = [
    0,
    316,
    652,
    1048,
    1500,
    2024,
    2724,
    3828,
    4956,
    6124,
    7244,
    8112,
    8696,
    9208,
    9612,
    9964,
    10300,
]


def prewarm_npu_causal_conv1d_once() -> None:
    """Run one NPU causal_conv1d call to initialize the operator path."""
    global _CONV1D_PREWARM_DONE

    if _CONV1D_PREWARM_DONE:
        return

    try:
        import torch

        if not hasattr(torch, "npu") or not hasattr(torch.ops, "npu"):
            return
        if not hasattr(torch.ops.npu, "npu_causal_conv1d"):
            return

        query_start_loc = _QUERY_START_LOC
        num_seqs = len(_QUERY_START_LOC) - 1
        device = f"npu:{torch.npu.current_device()}"

        x = torch.zeros((_SEQ_LEN, _DIM), dtype=torch.bfloat16, device=device).contiguous()
        weight = torch.zeros((_WIDTH, _DIM), dtype=torch.bfloat16, device=device).contiguous()
        conv_states = torch.zeros((num_seqs, _WIDTH - 1, _DIM), dtype=torch.bfloat16, device=device)

        with torch.no_grad():
            torch.ops.npu.npu_causal_conv1d(
                x=x,
                weight=weight,
                bias=None,
                conv_states=conv_states,
                query_start_loc=query_start_loc,
                cache_indices=list(range(num_seqs)),
                initial_state_mode=[0] * num_seqs,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=0,
            )
        _CONV1D_PREWARM_DONE = True
    except Exception:
        return
