from typing import List
import numpy as np


def print_bounds(lbs: List[float], ubs: List[float], prnt_name: str = "", num_print: int = 10) -> None:
    print(prnt_name, " lbs: ", lbs[:num_print])
    print(prnt_name, " ubs: ", ubs[:num_print])


def compare_arrays_diff_idx(arr1: List[float], arr2: List[float], arr_name: str = "") -> bool:
    assert len(arr1) == len(arr2)
    res_eq = True
    for i in range(len(arr1)):
        if not np.isclose(arr1[i], arr2[i]):
            print(arr_name, " index: ", i, ", val1: ", arr1[i], ", val2:", arr2[i])
            res_eq = False
    return res_eq
