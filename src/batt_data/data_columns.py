from typing import Dict, Optional

"""
    "*": Use same name as in raw data
    None: Don't import this column by default
"""

_DATA_COLUMNS_MAPPING: Dict[str, Optional[str]] = {
    "U_Battery": "U_Batt",
    "I_Battery": "I_Batt",
    "SOC_Battery": "SOC_Batt",
    "Temperature_1": "T_Cell_1_2",
    "Temperature_2": "T_Cell_3_4",
    "Temperature_3": "T_Cell_5_6",
    "Temperature_4": "T_Cell_7_8",
    "U_CR": None,
    "I_CR": None,
    "U_Cell_1": "*",
    "I_CNV_Cell_1": "*",
    "T_CNV_Cell_1": None,
    "SOC_Cell_1": None,
    "U_Cell_2": "*",
    "I_CNV_Cell_2": "*",
    "T_CNV_Cell_2": None,
    "SOC_Cell_2": None,
    "U_Cell_3": "*",
    "I_CNV_Cell_3": "*",
    "T_CNV_Cell_3": None,
    "SOC_Cell_3": None,
    "U_Cell_4": "*",
    "I_CNV_Cell_4": "*",
    "T_CNV_Cell_4": None,
    "SOC_Cell_4": None,
    "U_Cell_5": "*",
    "I_CNV_Cell_5": "*",
    "T_CNV_Cell_5": None,
    "SOC_Cell_5": None,
    "U_Cell_6": "*",
    "I_CNV_Cell_6": "*",
    "T_CNV_Cell_6": None,
    "SOC_Cell_6": None,
    "U_Cell_7": "*",
    "I_CNV_Cell_7": "*",
    "T_CNV_Cell_7": None,
    "SOC_Cell_7": None,
    "U_Cell_8": "*",
    "I_CNV_Cell_8": "*",
    "T_CNV_Cell_8": None,
    "SOC_Cell_8": None,
}

DATA_COLUMNS_MAPPING = {
    k: v if v != "*" else k for (k, v) in _DATA_COLUMNS_MAPPING.items()
}
