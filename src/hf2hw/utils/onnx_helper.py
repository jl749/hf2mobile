from typing import Set

import onnx
import onnx_ir as ir

# ── onnx/merge.py ───────────────────────────────────────────────────────────────────


def ensure_opset_imports(model: onnx.ModelProto, new_imports) -> None:
    """Add opset imports from a function into the model if not already present."""
    existing = {(o.domain, o.version) for o in model.opset_import}
    existing_domains = {o.domain for o in model.opset_import}
    for op in new_imports:
        if op.domain in existing_domains:
            continue  # do not clash with existing version pins
        if (op.domain, op.version) in existing:
            continue
        model.opset_import.append(op)


def update_opset(model: onnx.ModelProto, domain: str, version: int) -> None:
    """
    If `onnx.OperatorSetIdProto(domain, version)` does not exist under opset_import add it.
    If opset_import exists but the version is lower upgrade it.
    """
    for oi in model.opset_import:
        if oi.domain == domain:
            if oi.version < version:
                oi.version = version  # NOTE: version up the opset
            return
    model.opset_import.append(onnx.helper.make_opsetid(domain, version))


def drop_attributes(node: onnx.NodeProto, names_to_drop: Set[str]) -> None:
    """Drop AttributeProtos by its names from the passed NodeProto"""
    keep = [a for a in node.attribute if a.name not in names_to_drop]
    del node.attribute[:]
    node.attribute.extend(keep)


# ── onnx/fusion/ ────────────────────────────────────────────────────────────────────


def get_scalar(val: ir.Value) -> float | None:
    """Extract a scalar float from a constant-valued IR Value, or None."""
    try:
        cv = val.const_value
        if cv is None:
            return None
        return float(cv.numpy().flat[0])
    except Exception:
        return None
