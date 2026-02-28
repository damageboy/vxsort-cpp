import json

from bitonic_super_optimizer import PermutationGadget, SolutionNode


def export_solutions_to_json(
    roots: list[SolutionNode],
    output_path: str,
    *,
    natural_order: bool = False,
    vm_name: str,
    prim_type_name: str,
    num_vecs: int,
):
    """Generate JSON with all solutions and costs.

    The solution structure is a DAG (children are shared across nodes
    with the same output state).  To avoid exponential blowup during
    serialization, each unique node is emitted once in a flat ``nodes``
    dict keyed by a stable id, and children are referenced by id.

    Output format::

        {
          "roots": ["n0", "n1", ...],
          "nodes": {
            "n0": { "stage": ..., "children": ["n3", "n4"], ... },
            ...
          }
        }
    """

    def gadget_to_dict(gadget: PermutationGadget) -> dict:
        unified, top_out_idx, bottom_out_idx = gadget.unified_instructions()
        return {
            "top_instructions": [
                {"name": inst.intrinsic_name, "args": inst.args}
                for inst in gadget.top_instructions
            ],
            "bottom_instructions": [
                {"name": inst.intrinsic_name, "args": inst.args}
                for inst in gadget.bottom_instructions
            ],
            "unified_instructions": [
                {"name": inst.intrinsic_name, "args": inst.args} for inst in unified
            ],
            "top_output_index": top_out_idx,
            "bottom_output_index": bottom_out_idx,
        }

    # Assign stable ids and serialize each node exactly once.
    nodes_dict: dict[str, dict] = {}
    node_id_map: dict[int, str] = {}  # id(node) -> stable string id
    counter = 0

    def register_node(node: SolutionNode) -> str:
        nonlocal counter
        obj_id = id(node)
        if obj_id in node_id_map:
            return node_id_map[obj_id]
        stable_id = f"n{counter}"
        counter += 1
        node_id_map[obj_id] = stable_id
        # Register children first so their ids are available
        child_ids = [register_node(child) for child in node.children]
        nodes_dict[stable_id] = {
            "stage": node.stage,
            "input_state": {
                "top": node.input_state.top,
                "bottom": node.input_state.bottom,
            },
            "output_state": {
                "top": node.output_state.top,
                "bottom": node.output_state.bottom,
            },
            "gadgets": [gadget_to_dict(g) for g in node.gadgets],
            "gadget_count": len(node.gadgets),
            "children": child_ids,
        }
        return stable_id

    root_ids = [register_node(root) for root in roots]

    output = {
        "natural_order": natural_order,
        "vector_machine": vm_name,
        "primitive_type": prim_type_name,
        "num_vecs": num_vecs,
        "roots": root_ids,
        "nodes": nodes_dict,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(
        f"Exported {len(roots)} roots, {len(nodes_dict)} unique nodes to {output_path}"
    )
