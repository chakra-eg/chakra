import re


def has_backward_parent(op):
    if not op.parent or op.parent.id == op.id:  # Top op
        return False
    if is_backward_parent(op):
        return True
    return has_backward_parent(op.parent)


def is_backward_parent(op):
    return (
        "autograd::engine::evaluate_function: " in op.name
        or "Optimizer.step" in op.name
    )


def is_backward_aten(op):
    return op.name.startswith("aten::") and has_backward_parent(op)


def fbgemm_input_args_indices(n):
    idx_list = None
    if "sgd" in n.name or "adagrad" in n.name:
        # exact_sgd: 11: indices, 12: offsets, 14: indice_weights
        if n.inputs[14] == "<None>":
            idx_list = [11, 12]
        else:
            idx_list = [11, 12, 14]
    return idx_list


def is_fbgemm_forward(op):
    return "fbgemm::split_embedding_codegen_lookup_" in op.name


def is_fbgemm_backward(op):
    return "CppNode<SplitLookupFunction_" in op.name and not is_backward_parent(op)


# TODO: Hopefully merge is_fbgemm and skip_op, ignore tid == 2
def skip_op(op):
    # Workaround: skip bounds check indices and other ops under embedding lookup module
    return (
        not is_fbgemm_forward(op)
        and op.parent is not None
        and (
            "embedding_lookup" in op.parent.name
            or "param|SplitTableBatchedEmbeddingBagsCodegen" in op.parent.name
            or (
                "fbgemm::" in op.name
                and "fbgemm::split_embedding_codegen_lookup_" not in op.name
            )
        )
        or ("fused" in op.name)
        or (
            op.name
            in [
                "aten::empty",
                "aten::to",
                "aten::lift",
                "aten::detach_",
                "aten::set_",
                "aten::pin_memory",
            ]
            and "thread" in op.parent.name
            and op.tid == 2
        )
        or (op.name == "record_param_comms" and op.inputs[3] == "init")
        or (op.name == "aten::view" and "aten::view.dtype" in op.op_schema)
    )
