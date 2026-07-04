from typing import Set

import onnx


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


def drop_attributes(node: onnx.NodeProto, names_to_drop: Set[str]) -> None:
    keep = [a for a in node.attribute if a.name not in names_to_drop]
    del node.attribute[:]
    node.attribute.extend(keep)
