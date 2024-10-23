"""Naming convention for cells in column names of dataframes"""


def get_cell_tag(cellnr: int) -> str:
    if cellnr == -1:
        return "pack"
    else:
        return f"c{cellnr}"


def get_causal_tag(causal: bool) -> str:
    if causal:
        return "causal"
    else:
        return "acausal"
