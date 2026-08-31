# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.glm5next.nvidia import model as glm_model
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)


def _make_glm5next_model() -> Glm5NextModel:
    model = object.__new__(Glm5NextModel)
    torch.nn.Module.__init__(model)
    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 2)
    object.__setattr__(model, "is_sequence_parallel", False)
    object.__setattr__(model, "aux_hidden_state_layers", (1, 2))
    return model


def _patch_pp_group(monkeypatch) -> None:
    monkeypatch.setattr(
        glm_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )


def test_glm5next_advertises_eagle3_support():
    assert supports_eagle3(Glm5NextForCausalLM)
    assert supports_eagle3(Glm5NextForConditionalGeneration)


def test_glm5next_uses_shared_eagle3_layer_configuration():
    target = object.__new__(Glm5NextForCausalLM)
    torch.nn.Module.__init__(target)
    inner = object.__new__(Glm5NextModel)
    torch.nn.Module.__init__(inner)
    object.__setattr__(inner, "layers", [None] * 45)
    object.__setattr__(target, "model", inner)

    target.set_aux_hidden_state_layers((6, 15, 25, 34, 43))

    assert inner.aux_hidden_state_layers == (6, 15, 25, 34, 43)
    assert target.get_eagle3_default_aux_hidden_state_layers() == (2, 22, 42)


def test_glm5next_forward_extracts_aux_hidden_states_in_order(monkeypatch):
    model = _make_glm5next_model()
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    layer_outputs = (torch.tensor([[3.0, 4.0]]), torch.tensor([[5.0, 6.0]]))
    layer_residuals = (torch.tensor([[7.0, 8.0]]), torch.tensor([[9.0, 10.0]]))

    object.__setattr__(model, "norm", lambda h: h)
    object.__setattr__(
        model,
        "_active_layers",
        [
            Mock(return_value=(layer_outputs[0], layer_residuals[0], None, None)),
            Mock(return_value=(layer_outputs[1], layer_residuals[1], None, None)),
        ],
    )
    _patch_pp_group(monkeypatch)

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    # GLM layers return the post-residual output; the aux capture must not add
    # the residual channel a second time.
    torch.testing.assert_close(output, layer_outputs[1])
    torch.testing.assert_close(aux_hidden_states[0], layer_outputs[0])
    torch.testing.assert_close(aux_hidden_states[1], layer_outputs[1])


def test_glm5next_forward_without_aux_layers_returns_tensor(monkeypatch):
    model = _make_glm5next_model()
    object.__setattr__(model, "aux_hidden_state_layers", ())
    object.__setattr__(model, "norm", lambda h: h)
    object.__setattr__(
        model,
        "_active_layers",
        [Mock(return_value=(torch.tensor([[3.0, 4.0]]), None, None, None))],
    )
    _patch_pp_group(monkeypatch)

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.tensor([[1.0, 2.0]]),
    )

    assert isinstance(output, torch.Tensor)


def test_glm5next_forward_gathers_aux_under_sequence_parallel(monkeypatch):
    model = _make_glm5next_model()
    object.__setattr__(model, "aux_hidden_state_layers", (1,))
    object.__setattr__(model, "is_sequence_parallel", True)
    layer_hidden = torch.tensor([[3.0, 4.0]])
    object.__setattr__(model, "norm", lambda h: h)
    object.__setattr__(
        model,
        "_active_layers",
        [Mock(return_value=(layer_hidden, None, None, None))],
    )
    _patch_pp_group(monkeypatch)
    monkeypatch.setattr(glm_model, "sp_shard", lambda h: h)
    monkeypatch.setattr(glm_model, "sp_all_gather", lambda h: h)

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.tensor([[1.0, 2.0]]),
    )

    torch.testing.assert_close(output, layer_hidden)
    torch.testing.assert_close(aux_hidden_states[0], layer_hidden)
