from pathlib import Path

import torch
import torch.nn as nn


def export_model(
    model: nn.Module,
    example_input,
    output_path: Path,
    dynamic_shapes=None,
) -> None:
    model.eval()
    with torch.no_grad():
        exported_program = torch.export.export(
            model,
            args=(example_input,),
            dynamic_shapes=dynamic_shapes,
        )
    torch.export.save(exported_program, output_path)


def verify_exported(
    original_model: nn.Module,
    artifact_path: Path,
    example_input,
    name: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    original_model.eval()
    ep = torch.export.load(artifact_path)
    with torch.no_grad():
        original_out = original_model(example_input)
        exported_out = ep.module()(example_input)
    if not torch.allclose(original_out, exported_out, atol=atol, rtol=rtol):
        max_diff = (original_out - exported_out).abs().max().item()
        raise AssertionError(
            f"[FAIL] {name}: max absolute diff {max_diff:.2e} exceeds atol={atol}"
        )
    print(f"[PASS] {name}: outputs match (atol={atol}, rtol={rtol})")


def verify_exported_dict(
    original_model: nn.Module,
    artifact_path: Path,
    example_input,
    name: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Verify exported model whose forward returns a dict of tensors."""
    original_model.eval()
    ep = torch.export.load(artifact_path)
    with torch.no_grad():
        original_out = original_model(example_input)
        exported_out = ep.module()(example_input)
    for key in original_out:
        if not torch.allclose(
            original_out[key], exported_out[key], atol=atol, rtol=rtol
        ):
            max_diff = (original_out[key] - exported_out[key]).abs().max().item()
            raise AssertionError(
                f"[FAIL] {name}[{key}]: max absolute diff {max_diff:.2e} exceeds atol={atol}"
            )
    print(f"[PASS] {name}: all outputs match (atol={atol}, rtol={rtol})")
