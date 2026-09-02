# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ============================================================
# 1. CHEMICAL DATABASE
# ============================================================

CHEMICAL_DATABASE = {

    # --------------------------------------------------------
    # PRECIPITANTS
    # --------------------------------------------------------

    "Oxalate": {
        "pkas": [1.25, 4.27],
        "charge_anion": 2,
        "metals": {

            "Ag+": {"ksp": 3.5e-11, "x": 2, "y": 1},
            "Ba2+": {"ksp": 1.6e-6, "x": 1, "y": 1},
            "Bi3+": {"ksp": 1.0e-24, "x": 2, "y": 3},
            "Ca2+": {"ksp": 2.3e-9, "x": 1, "y": 1},
            "Cd2+": {"ksp": 1.4e-8, "x": 1, "y": 1},
            "Ce3+": {"ksp": 3.0e-26, "x": 2, "y": 3},
            "Ce4+": {"ksp": 1.0e-29, "x": 1, "y": 2},
            "Co2+": {"ksp": 6.3e-8, "x": 1, "y": 1},
            "Cu2+": {"ksp": 4.4e-10, "x": 1, "y": 1},
            "Dy3+": {"ksp": 2.0e-26, "x": 2, "y": 3},
            "Er3+": {"ksp": 2.2e-26, "x": 2, "y": 3},
            "Eu3+": {"ksp": 1.8e-26, "x": 2, "y": 3},
            "Fe2+": {"ksp": 3.2e-7, "x": 1, "y": 1},
            "Ga3+": {"ksp": 1.0e-21, "x": 2, "y": 3},
            "Gd3+": {"ksp": 2.1e-26, "x": 2, "y": 3},
            "Hg2_2+": {"ksp": 1.8e-13, "x": 1, "y": 1},
            "Hg2+": {"ksp": 1.5e-7, "x": 1, "y": 1},
            "Ho3+": {"ksp": 2.4e-26, "x": 2, "y": 3},
            "In3+": {"ksp": 5.8e-24, "x": 2, "y": 3},
            "La3+": {"ksp": 2.0e-26, "x": 2, "y": 3},
            "Lu3+": {"ksp": 3.2e-26, "x": 2, "y": 3},
            "Mg2+": {"ksp": 8.6e-5, "x": 1, "y": 1},
            "Mn2+": {"ksp": 1.1e-15, "x": 1, "y": 1},
            "Nd3+": {"ksp": 3.2e-26, "x": 2, "y": 3},
            "Ni2+": {"ksp": 4.0e-10, "x": 1, "y": 1},
            "Pb2+": {"ksp": 8.5e-9, "x": 1, "y": 1},
            "Pr3+": {"ksp": 2.8e-26, "x": 2, "y": 3},
            "Sc3+": {"ksp": 1.0e-25, "x": 2, "y": 3},
            "Sm3+": {"ksp": 2.5e-26, "x": 2, "y": 3},
            "Sn2+": {"ksp": 1.0e-9, "x": 1, "y": 1},
            "Sn4+": {"ksp": 1.0e-30, "x": 1, "y": 2},
            "Sr2+": {"ksp": 5.6e-8, "x": 1, "y": 1},
            "Tb3+": {"ksp": 2.2e-26, "x": 2, "y": 3},
            "Th4+": {"ksp": 5.0e-25, "x": 1, "y": 2},
            "Tl+": {"ksp": 2.0e-4, "x": 2, "y": 1},
            "Tl3+": {"ksp": 1.0e-28, "x": 2, "y": 3},
            "Tm3+": {"ksp": 2.6e-26, "x": 2, "y": 3},
            "UO2_2+": {"ksp": 1.3e-8, "x": 1, "y": 1},
            "U4+": {"ksp": 1.0e-22, "x": 1, "y": 2},
            "Y3+": {"ksp": 5.0e-26, "x": 2, "y": 3},
            "Yb3+": {"ksp": 3.0e-26, "x": 2, "y": 3},
            "Zn2+": {"ksp": 2.8e-8, "x": 1, "y": 1},
        }
    },


    "Phosphate": {
        "pkas": [2.15, 7.20, 12.35],
        "charge_anion": 3,
        "metals": {

            "Ag+": {"ksp": 1.4e-16, "x": 3, "y": 1},
            "Al3+": {"ksp": 9.8e-21, "x": 1, "y": 1},
            "Ba2+": {"ksp": 3.4e-23, "x": 3, "y": 2},
            "Be2+": {"ksp": 1.5e-39, "x": 3, "y": 2},
            "Bi3+": {"ksp": 1.3e-23, "x": 1, "y": 1},
            "Ca2+": {"ksp": 2.1e-33, "x": 3, "y": 2},
            "Cd2+": {"ksp": 2.5e-33, "x": 3, "y": 2},
            "Ce3+": {"ksp": 1.0e-23, "x": 1, "y": 1},
            "Ce4+": {"ksp": 1.0e-64, "x": 3, "y": 4},
            "Co2+": {"ksp": 2.0e-35, "x": 3, "y": 2},
            "Cr3+": {"ksp": 2.4e-23, "x": 1, "y": 1},
            "Cu2+": {"ksp": 1.4e-37, "x": 3, "y": 2},
            "Dy3+": {"ksp": 4.0e-25, "x": 1, "y": 1},
            "Er3+": {"ksp": 3.2e-25, "x": 1, "y": 1},
            "Eu3+": {"ksp": 2.5e-25, "x": 1, "y": 1},
            "Fe2+": {"ksp": 1.0e-36, "x": 3, "y": 2},
            "Fe3+": {"ksp": 1.3e-22, "x": 1, "y": 1},
            "Ga3+": {"ksp": 1.0e-21, "x": 1, "y": 1},
            "Gd3+": {"ksp": 3.0e-25, "x": 1, "y": 1},
            "Hf4+": {"ksp": 1.0e-62, "x": 3, "y": 4},
            "Hg2_2+": {"ksp": 1.0e-61, "x": 3, "y": 2},
            "Hg2+": {"ksp": 8.8e-33, "x": 3, "y": 2},
            "Ho3+": {"ksp": 3.5e-25, "x": 1, "y": 1},
            "In3+": {"ksp": 1.0e-22, "x": 1, "y": 1},
            "La3+": {"ksp": 3.7e-23, "x": 1, "y": 1},
            "Lu3+": {"ksp": 5.0e-25, "x": 1, "y": 1},
            "Mg2+": {"ksp": 1.0e-24, "x": 3, "y": 2},
            "Mn2+": {"ksp": 1.0e-22, "x": 3, "y": 2},
            "Nb5+": {"ksp": 1.0e-85, "x": 3, "y": 5},
            "Nd3+": {"ksp": 4.0e-25, "x": 1, "y": 1},
            "Ni2+": {"ksp": 5.0e-31, "x": 3, "y": 2},
            "Pb2+": {"ksp": 7.9e-44, "x": 3, "y": 2},
            "Pr3+": {"ksp": 3.5e-23, "x": 1, "y": 1},
            "Sc3+": {"ksp": 1.0e-24, "x": 1, "y": 1},
            "Sm3+": {"ksp": 2.8e-25, "x": 1, "y": 1},
            "Sn2+": {"ksp": 1.0e-27, "x": 3, "y": 2},
            "Sn4+": {"ksp": 1.0e-60, "x": 3, "y": 4},
            "Sr2+": {"ksp": 4.0e-28, "x": 3, "y": 2},
            "Ta5+": {"ksp": 1.0e-88, "x": 3, "y": 5},
            "Tb3+": {"ksp": 3.2e-25, "x": 1, "y": 1},
            "Th4+": {"ksp": 1.0e-78, "x": 3, "y": 4},
            "Ti4+": {"ksp": 1.0e-65, "x": 3, "y": 4},
            "Tl3+": {"ksp": 1.0e-35, "x": 1, "y": 1},
            "Tm3+": {"ksp": 4.2e-25, "x": 1, "y": 1},
            "U4+": {"ksp": 1.0e-80, "x": 3, "y": 4},
            "V3+": {"ksp": 1.0e-22, "x": 1, "y": 1},
            "Y3+": {"ksp": 1.0e-25, "x": 1, "y": 1},
            "Yb3+": {"ksp": 4.5e-25, "x": 1, "y": 1},
            "Zn2+": {"ksp": 9.0e-33, "x": 3, "y": 2},
            "Zr4+": {"ksp": 1.0e-62, "x": 3, "y": 4},
        }
    },


    "Carbonate": {
        "pkas": [6.35, 10.33],
        "charge_anion": 2,
        "metals": {

            "Ag+": {"ksp": 8.5e-12, "x": 2, "y": 1},
            "Ba2+": {"ksp": 2.6e-9, "x": 1, "y": 1},
            "Be2+": {"ksp": 2.0e-10, "x": 1, "y": 1},
            "Ca2+": {"ksp": 4.5e-9, "x": 1, "y": 1},
            "Cd2+": {"ksp": 1.0e-12, "x": 1, "y": 1},
            "Ce4+": {"ksp": 1.0e-36, "x": 1, "y": 2},
            "Co2+": {"ksp": 1.0e-10, "x": 1, "y": 1},
            "Cu2+": {"ksp": 1.4e-10, "x": 1, "y": 1},
            "Fe2+": {"ksp": 3.1e-11, "x": 1, "y": 1},
            "Hg2_2+": {"ksp": 3.6e-17, "x": 1, "y": 1},
            "Hg2+": {"ksp": 3.0e-13, "x": 1, "y": 1},
            "Mg2+": {"ksp": 3.5e-8, "x": 1, "y": 1},
            "Mn2+": {"ksp": 2.2e-11, "x": 1, "y": 1},
            "Ni2+": {"ksp": 1.3e-7, "x": 1, "y": 1},
            "Pb2+": {"ksp": 7.4e-14, "x": 1, "y": 1},
            "Sn2+": {"ksp": 1.0e-20, "x": 1, "y": 1},
            "Sn4+": {"ksp": 1.0e-38, "x": 1, "y": 2},
            "Sr2+": {"ksp": 5.6e-10, "x": 1, "y": 1},
            "Th4+": {"ksp": 1.0e-35, "x": 1, "y": 2},
            "Tl+": {"ksp": 4.3e-4, "x": 2, "y": 1},
            "U4+": {"ksp": 1.0e-36, "x": 1, "y": 2},
            "UO2_2+": {"ksp": 3.0e-14, "x": 1, "y": 1},
            "VO2+": {"ksp": 1.0e-18, "x": 1, "y": 1},
            "Zn2+": {"ksp": 1.2e-10, "x": 1, "y": 1},
        }
    },


    # --------------------------------------------------------
    # COMPLEXANTS
    # --------------------------------------------------------

    "EDTA": {
        "pkas": [0.0, 1.5, 2.0, 2.66, 6.16, 10.24],
        "log_betas": {

            "Ag+": [7.3],
            "Al3+": [16.1],
            "Ba2+": [7.9],
            "Be2+": [9.2],
            "Bi3+": [27.8],
            "Ca2+": [10.6],
            "Cd2+": [16.5],
            "Ce3+": [16.0],
            "Ce4+": [24.4],
            "Co2+": [16.3],
            "Cr3+": [23.4],
            "Cu2+": [18.8],
            "Dy3+": [18.3],
            "Er3+": [18.9],
            "Eu3+": [17.3],
            "Fe2+": [14.3],
            "Fe3+": [25.1],
            "Ga3+": [20.3],
            "Gd3+": [17.4],
            "Hf4+": [29.5],
            "Hg2_2+": [0.0],
            "Hg2+": [21.8],
            "Ho3+": [18.6],
            "In3+": [25.0],
            "La3+": [15.5],
            "Lu3+": [19.8],
            "Mg2+": [8.7],
            "Mn2+": [13.8],
            "Nb5+": [0.0],
            "Nd3+": [16.6],
            "Ni2+": [18.6],
            "Pb2+": [18.0],
            "Pr3+": [16.4],
            "Sc3+": [23.1],
            "Sm3+": [17.1],
            "Sn2+": [22.1],
            "Sn4+": [0.0],
            "Sr2+": [8.7],
            "Ta5+": [0.0],
            "Tb3+": [17.9],
            "Th4+": [23.2],
            "Ti4+": [17.3],
            "Tl+": [2.2],
            "Tl3+": [22.5],
            "Tm3+": [19.3],
            "U4+": [25.8],
            "UO2_2+": [10.2],
            "V3+": [26.0],
            "VO2+": [18.8],
            "Y3+": [18.1],
            "Yb3+": [19.5],
            "Zn2+": [16.5],
            "Zr4+": [29.5],
        }
    },


    "Glycine": {
        "pkas": [2.34, 9.60],
        "log_betas": {

            "Ag+": [3.4, 6.9],
            "Al3+": [2.1, 4.0, 5.2],
            "Ba2+": [0.8],
            "Be2+": [1.4, 2.5],
            "Bi3+": [0.0],
            "Ca2+": [1.4, 2.3],
            "Cd2+": [4.3, 7.8, 10.1],
            "Ce3+": [3.2, 5.8, 7.8],
            "Ce4+": [0.0],
            "Co2+": [4.3, 7.9, 10.8],
            "Cr3+": [5.1, 9.8, 13.9],
            "Cu2+": [8.2, 15.1],
            "Dy3+": [3.5, 6.4, 8.8],
            "Er3+": [3.6, 6.5, 9.0],
            "Eu3+": [3.4, 6.2, 8.5],
            "Fe2+": [4.3, 7.7, 10.0],
            "Fe3+": [0.0],
            "Ga3+": [4.2, 8.0, 11.1],
            "Gd3+": [3.4, 6.1, 8.4],
            "Hf4+": [0.0],
            "Hg2_2+": [0.0],
            "Hg2+": [10.3, 19.2],
            "Ho3+": [3.5, 6.4, 8.9],
            "In3+": [5.7, 10.7, 14.8],
            "La3+": [3.0, 5.4, 7.5],
            "Lu3+": [3.8, 6.9, 9.6],
            "Mg2+": [2.2, 3.9],
            "Mn2+": [2.8, 5.5, 8.2],
            "Nb5+": [0.0],
            "Nd3+": [3.2, 5.9, 8.2],
            "Ni2+": [5.4, 9.7, 14.2],
            "Pb2+": [4.8, 7.8],
            "Pr3+": [3.1, 5.6, 7.9],
            "Sc3+": [4.5, 8.4, 11.5],
            "Sm3+": [3.3, 6.0, 8.3],
            "Sn2+": [4.5, 8.1],
            "Sn4+": [0.0],
            "Sr2+": [1.0],
            "Ta5+": [0.0],
            "Tb3+": [3.4, 6.2, 8.6],
            "Th4+": [0.0],
            "Ti4+": [0.0],
            "Tl+": [0.9],
            "Tl3+": [6.5, 11.8, 15.5],
            "Tm3+": [3.7, 6.7, 9.2],
            "U4+": [0.0],
            "UO2_2+": [2.6, 4.4],
            "V3+": [0.0],
            "VO2+": [3.2, 6.1],
            "Y3+": [3.5, 6.3, 8.6],
            "Yb3+": [3.8, 6.8, 9.4],
            "Zn2+": [5.0, 9.3, 12.1],
        }
    },


    "Citrate": {
        "pkas": [3.13, 4.76, 6.40],
        "log_betas": {

            "Ag+": [0.9],
            "Al3+": [8.1, 14.2],
            "Ba2+": [2.5],
            "Be2+": [3.3],
            "Bi3+": [11.2],
            "Ca2+": [3.5],
            "Cd2+": [4.2, 6.5],
            "Ce3+": [7.5, 12.1],
            "Ce4+": [13.5],
            "Co2+": [5.0, 7.8],
            "Cr3+": [9.3],
            "Cu2+": [5.9, 10.2],
            "Dy3+": [7.9, 13.0],
            "Er3+": [8.1, 13.3],
            "Eu3+": [7.8, 12.8],
            "Fe2+": [4.4, 7.1],
            "Fe3+": [11.5, 18.5],
            "Ga3+": [10.0],
            "Gd3+": [7.8, 12.7],
            "Hf4+": [17.5],
            "Hg2_2+": [0.0],
            "Hg2+": [10.9],
            "Ho3+": [8.0, 13.2],
            "In3+": [10.2, 16.5],
            "La3+": [7.2, 11.5],
            "Lu3+": [8.5, 14.1],
            "Mg2+": [3.4],
            "Mn2+": [3.7, 5.8],
            "Nb5+": [0.0],
            "Nd3+": [7.6, 12.3],
            "Ni2+": [5.4, 8.4],
            "Pb2+": [5.5, 8.3],
            "Pr3+": [7.4, 11.9],
            "Sc3+": [10.5],
            "Sm3+": [7.7, 12.6],
            "Sn2+": [5.5],
            "Sn4+": [0.0],
            "Sr2+": [2.8],
            "Ta5+": [0.0],
            "Tb3+": [7.9, 12.9],
            "Th4+": [13.0, 22.2],
            "Ti4+": [14.2],
            "Tl+": [1.0],
            "Tl3+": [12.5],
            "Tm3+": [8.2, 13.5],
            "U4+": [14.5],
            "UO2_2+": [6.4, 11.1],
            "V3+": [7.8],
            "VO2+": [6.8, 11.8],
            "Y3+": [7.8, 12.8],
            "Yb3+": [8.4, 13.9],
            "Zn2+": [5.0, 8.6],
        }
    },


    "Oxalate Complex": {
        "pkas": [1.25, 4.27],
        "log_betas": {

            "Al3+": [6.1, 11.1, 15.1],
            "Be2+": [1.9, 3.2],
            "Cr3+": [5.3, 10.5, 15.2],
            "Fe3+": [7.5, 13.6, 18.5],
            "Hf4+": [10.2, 19.5, 27.5],
            "Nb5+": [9.5, 18.0, 25.0],
            "Ta5+": [9.0, 17.5, 24.0],
            "Ti4+": [8.5, 16.0],
            "V3+": [6.2, 11.5, 15.5],
            "VO2+": [6.3, 11.4],
            "Zr4+": [10.0, 19.2, 27.2],
        }
    }
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def calculate_alpha_inverse(pH, pkas):
 
    h = 10.0 ** (-pH)
    kas = [10.0 ** (-pka) for pka in pkas]

    alpha_inverse = 1.0
    cumulative_product = 1.0

    for i, ka in enumerate(kas[::-1]):

        cumulative_product *= ka

        alpha_inverse += (
            (h ** (i + 1))
            / cumulative_product
        )

    return alpha_inverse


def get_complex_parameters(complexant_name, metal_name):
 
    if complexant_name is None:
        return {
            "active": False,
            "status": "No complexant",
            "N": 0,
            "beta": 0.0,
            "log_beta": None
        }

    complexant = CHEMICAL_DATABASE[complexant_name]

    if metal_name not in complexant["log_betas"]:
        return {
            "active": False,
            "status": "No data",
            "N": 0,
            "beta": 0.0,
            "log_beta": None
        }

    beta_series = complexant["log_betas"][metal_name]

    if len(beta_series) == 0:
        return {
            "active": False,
            "status": "No data",
            "N": 0,
            "beta": 0.0,
            "log_beta": None
        }

    log_beta = beta_series[-1]
    N = len(beta_series)

    if log_beta <= 0:
        return {
            "active": False,
            "status": "No stable complex represented",
            "N": 0,
            "beta": 0.0,
            "log_beta": log_beta
        }

    return {
        "active": True,
        "status": "Data available",
        "N": N,
        "beta": 10.0 ** log_beta,
        "log_beta": log_beta
    }


def build_metal_system(
    selected_metals,
    concentrations,
    precipitant_name,
    complexant_name
):

    precipitant_data = CHEMICAL_DATABASE[
        precipitant_name
    ]["metals"]

    systems = []

    for metal in selected_metals:

        if metal not in precipitant_data:
            continue

        precip_data = precipitant_data[metal]

        complex_data = get_complex_parameters(
            complexant_name,
            metal
        )

        systems.append({

            "name": metal,

            "M_initial": concentrations[metal],

            "Ksp": precip_data["ksp"],
            "x": precip_data["x"],
            "y": precip_data["y"],

            "complex_active":
                complex_data["active"],

            "complex_status":
                complex_data["status"],

            "N":
                complex_data["N"],

            "Beta":
                complex_data["beta"],

            "log_beta":
                complex_data["log_beta"]
        })

    return systems


# ============================================================
# 3. CORE EQUILIBRIUM ENGINE
# ============================================================

LOG_FREE_SPECIES_FLOOR = -300.0

BISECTION_ITERATIONS = 60


def _equilibrium_free_metal_ceiling(
    ksp,
    x,
    y,
    A_free
):

    A_free = max(
        A_free,
        1e-300
    )

    log_ceiling = (

        np.log10(ksp)
        - y * np.log10(A_free)

    ) / x

    if log_ceiling > 300.0:
        return np.inf

    return 10.0 ** log_ceiling


def _metal_free_and_soluble(
    m,
    M_ceiling,
    Y_free
):

    if m["complex_active"]:

        complex_term = (
            m["Beta"]
            * (Y_free ** m["N"])
        )

    else:

        complex_term = 0.0

    M_from_complexation = (

        m["M_initial"]
        / (1.0 + complex_term)

    )

    M_free = min(

        M_from_complexation,
        M_ceiling

    )

    M_soluble = (

        M_free
        * (1.0 + complex_term)

    )

    M_soluble = min(

        m["M_initial"],
        M_soluble

    )

    M_precipitated = max(

        0.0,
        m["M_initial"] - M_soluble

    )

    return (
        M_free,
        M_soluble,
        M_precipitated
    )


def _bracket_log(
    residual_fn,
    log_low,
    log_high,
    scan_points=80
):

    us = np.linspace(log_low, log_high, scan_points)

    prev_u = us[0]
    prev_r = residual_fn(10.0 ** prev_u)

    if prev_r >= 0:
        return log_low, log_low

    for u in us[1:]:

        r = residual_fn(10.0 ** u)

        if r >= 0:
            return prev_u, u

        prev_u, prev_r = u, r

    return prev_u, log_high


def _bisect_log(
    residual_fn,
    log_high,
    iterations=BISECTION_ITERATIONS
):

    low_u, high_u = _bracket_log(
        residual_fn,
        LOG_FREE_SPECIES_FLOOR,
        log_high
    )

    for _ in range(iterations):

        mid_u = 0.5 * (
            low_u + high_u
        )

        if residual_fn(
            10.0 ** mid_u
        ) < 0:

            low_u = mid_u

        else:

            high_u = mid_u

    return 10.0 ** (
        0.5 * (
            low_u + high_u
        )
    )


def solve_equilibrium(
    pH,
    metals,
    complexant_total,
    precipitant_total,
    complexant_pkas,
    precipitant_pkas
):

    if len(metals) == 0:

        return {}


    complexant_active = (
        complexant_total > 0.0
    )


    alpha_Y_inv = (

        calculate_alpha_inverse(
            pH,
            complexant_pkas
        )

        if complexant_active

        else 1.0
    )


    alpha_A_inv = (
        calculate_alpha_inverse(
            pH,
            precipitant_pkas
        )
    )


    # --------------------------------------------------------
    # No precipitant
    # --------------------------------------------------------

    if precipitant_total <= 0.0:

        return {
            m["name"]: 0.0
            for m in metals
        }


    # --------------------------------------------------------
    # Inner solve:
    # free complexant
    # --------------------------------------------------------

    def calculate_free_complexant(
        A_free
    ):

        if not complexant_active:

            return 0.0


        high_y = (
            complexant_total
            / alpha_Y_inv
        )


        if high_y <= 0.0:

            return 0.0


        # M_ceiling depends only on A_free (fixed for this call),
        # never on the Y_free being solved for below. Computing it
        # once here — instead of once per bisection step — removes
        # ~60x redundant work per pH/concentration point.
        ceilings = {

            m["name"]: _equilibrium_free_metal_ceiling(
                m["Ksp"], m["x"], m["y"], A_free
            )

            for m in metals

        }


        def residual(
            Y_free
        ):

            calculated_complexant = (
                Y_free
                * alpha_Y_inv
            )


            for m in metals:

                if not m["complex_active"]:

                    continue


                M_free, _, _ = (
                    _metal_free_and_soluble(
                        m,
                        ceilings[m["name"]],
                        Y_free
                    )
                )


                complexed_metal = (

                    m["Beta"]
                    * M_free
                    * (Y_free ** m["N"])

                )


                calculated_complexant += (

                    m["N"]
                    * complexed_metal

                )


            return (
                calculated_complexant
                - complexant_total
            )


        return _bisect_log(

            residual,
            np.log10(high_y)

        )


    # --------------------------------------------------------
    # Outer solve:
    # precipitant mass balance
    # --------------------------------------------------------

    def precipitant_balance(
        A_free
    ):

        Y_free = (
            calculate_free_complexant(
                A_free
            )
        )


        calculated_precipitant = (

            A_free
            * alpha_A_inv

        )


        for m in metals:

            M_ceiling = (
                _equilibrium_free_metal_ceiling(
                    m["Ksp"], m["x"], m["y"], A_free
                )
            )

            _, _, M_precipitated = (

                _metal_free_and_soluble(
                    m,
                    M_ceiling,
                    Y_free
                )

            )


            calculated_precipitant += (

                (m["y"] / m["x"])
                * M_precipitated

            )


        return (

            calculated_precipitant
            - precipitant_total

        )


    upper_bound_A = max(

        precipitant_total
        / alpha_A_inv,

        1e-300

    )


    A_free_actual = _bisect_log(

        precipitant_balance,

        np.log10(
            upper_bound_A
        )

    )


    # --------------------------------------------------------
    # Final self-consistent solution
    # --------------------------------------------------------

    Y_free_final = (
        calculate_free_complexant(
            A_free_actual
        )
    )


    results = {}


    for m in metals:

        M_ceiling_final = (
            _equilibrium_free_metal_ceiling(
                m["Ksp"], m["x"], m["y"], A_free_actual
            )
        )

        _, M_soluble, _ = (
            _metal_free_and_soluble(
                m,
                M_ceiling_final,
                Y_free_final
            )
        )


        if m["M_initial"] > 0:

            precipitation_pct = (

                (
                    m["M_initial"]
                    - M_soluble
                )
                / m["M_initial"]

            ) * 100.0

        else:

            precipitation_pct = 0.0


        results[m["name"]] = float(
            np.clip(
                precipitation_pct,
                0.0,
                100.0
            )
        )


    return results


# ============================================================
# 4. STREAMLIT INTERFACE
# ============================================================

st.set_page_config(

    page_title=
        "Selective Precipitation Simulator",

    page_icon="🧪",

    layout="wide"

)


st.title(
    "🧪 Competitive metal precipitation in the presence of a complexing agent"
)


st.info(
    "The model uses one effective global metal–complex species "
    "per metal/complexant pair, following the highest available "
    "cumulative log β value in the database."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "System Configuration"
)


# ------------------------------------------------------------
# PRECIPITANT
# ------------------------------------------------------------

precipitant_name = st.sidebar.selectbox(

    "Precipitating agent",

    [
        "Oxalate",
        "Phosphate",
        "Carbonate"
    ]

)


# ------------------------------------------------------------
# COMPLEXANT
# ------------------------------------------------------------

# Oxalate Complex is deliberately removed from the available
# complexant options whenever Oxalate is selected as the
# precipitant. Both names represent the same real chemical species,
# so allowing both roles simultaneously would require two independent
# oxalate pools and would violate the intended mass-balance model.

complexant_options = [

    "None",
    "EDTA",
    "Glycine",
    "Citrate",
    "Oxalate Complex"

]


if precipitant_name == "Oxalate":

    complexant_options = [

        option
        for option in complexant_options

        if option != "Oxalate Complex"

    ]


complexant_name = st.sidebar.selectbox(

    "Complexing agent",

    complexant_options

)


use_complexant = (
    complexant_name != "None"
)


if precipitant_name == "Oxalate":

    st.sidebar.caption(
        "Oxalate Complex is unavailable when oxalate "
        "is selected as the precipitating agent because "
        "both entries represent the same chemical species."
    )


# ============================================================
# AVAILABLE METALS
# ============================================================

available_metals = list(

    CHEMICAL_DATABASE[
        precipitant_name
    ]["metals"].keys()

)


st.sidebar.subheader(
    "Metals"
)


selected_metals = st.sidebar.multiselect(

    "Select metals",

    available_metals,

    default=available_metals[:4]

)


if len(selected_metals) == 0:

    st.warning(
        "Please select at least one metal."
    )

    st.stop()


# ============================================================
# METAL CONCENTRATIONS
# ============================================================

st.sidebar.subheader(
    "Initial Metal Concentrations"
)


concentrations = {}


for metal in selected_metals:

    concentrations[metal] = (
        st.sidebar.number_input(

            f"{metal} concentration",

            min_value=0.0,

            value=0.05,

            step=0.01,

            format="%.6f",

            key=f"conc_{metal}"

        )
    )


# ============================================================
# COMPLEXANT SETTINGS
# ============================================================

if use_complexant:

    st.sidebar.subheader(
        "Complexant Settings"
    )


    complexant_total = (
        st.sidebar.number_input(

            "Total complexant concentration",

            min_value=0.0,

            value=0.10,

            step=0.01,

            format="%.6f"

        )
    )


    complexant_pkas = (
        CHEMICAL_DATABASE[
            complexant_name
        ]["pkas"]
    )


else:

    complexant_total = 0.0

    complexant_pkas = []


# ============================================================
# PRECIPITANT SETTINGS
# ============================================================

st.sidebar.subheader(
    "Precipitant Settings"
)


precipitant_total = (
    st.sidebar.number_input(

        "Total precipitant concentration",

        min_value=0.0,

        value=0.70,

        step=0.01,

        format="%.6f"

    )
)


precipitant_pkas = (
    CHEMICAL_DATABASE[
        precipitant_name
    ]["pkas"]
)


# ============================================================
# GRAPH SETTINGS
# ============================================================

st.sidebar.subheader(
    "pH Settings"
)


pH_min, pH_max = st.sidebar.slider(

    "pH range",

    min_value=0.0,

    max_value=14.0,

    value=(0.0, 14.0),

    step=0.5

)


fixed_pH = st.sidebar.slider(

    "Fixed pH for concentration graph",

    min_value=0.0,

    max_value=14.0,

    value=7.0,

    step=0.1

)


# ============================================================
# NUMBER OF POINTS
# ============================================================

st.sidebar.subheader(
    "Simulation Resolution"
)


pH_points = st.sidebar.slider(

    "pH graph points",

    100,

    1000,

    500,

    step=100

)


concentration_points = st.sidebar.slider(

    "Concentration graph points",

    50,

    500,

    200,

    step=50

)


# ============================================================
# BUILD SYSTEM
# ============================================================

metals = build_metal_system(

    selected_metals,

    concentrations,

    precipitant_name,

    complexant_name
    if use_complexant
    else None

)


# ============================================================
# DATABASE INFORMATION
# ============================================================

with st.expander(
    "View selected chemical parameters"
):

    parameter_rows = []


    for m in metals:

        parameter_rows.append({

            "Metal":
                m["name"],

            "Initial concentration":
                m["M_initial"],

            "Ksp":
                f"{m['Ksp']:.3e}",

            "x":
                m["x"],

            "y":
                m["y"],

            "Complex":
                (
                    "Yes"
                    if m["complex_active"]
                    else "No"
                ),

            "Complexation data":
                m["complex_status"],

            "N":
                (
                    m["N"]
                    if m["complex_active"]
                    else "-"
                ),

            "log β":
                (
                    m["log_beta"]
                    if m["complex_active"]
                    else "-"
                )

        })


    st.dataframe(

        pd.DataFrame(
            parameter_rows
        ),

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# GRAPH 1 — PRECIPITATION VS pH
# ============================================================

st.header(
    "1. Precipitation as a Function of pH"
)


pH_values = np.linspace(

    pH_min,

    pH_max,

    pH_points

)


pH_results = []


progress_bar = st.progress(0)


for i, pH in enumerate(
    pH_values
):

    equilibrium = solve_equilibrium(

        pH=pH,

        metals=metals,

        complexant_total=
            complexant_total,

        precipitant_total=
            precipitant_total,

        complexant_pkas=
            complexant_pkas,

        precipitant_pkas=
            precipitant_pkas

    )


    row = {
        "pH": pH
    }


    for metal in selected_metals:

        row[metal] = equilibrium.get(

            metal,

            0.0

        )


    pH_results.append(
        row
    )


    progress_bar.progress(

        int(

            (i + 1)
            / len(pH_values)
            * 100

        )

    )


progress_bar.empty()


df_pH = pd.DataFrame(
    pH_results
)


# ------------------------------------------------------------
# VISUAL SMOOTHING ONLY
# ------------------------------------------------------------

def _smooth_for_plot(series):

    n = len(series)

    window = max(5, min(31, (n // 20) | 1))

    if n <= window:
        return series

    try:

        smoothed = savgol_filter(
            series,
            window_length=window,
            polyorder=2
        )

        return np.clip(smoothed, 0, 100)

    except Exception:

        return series


df_pH_plot = df_pH.copy()

for metal in selected_metals:

    df_pH_plot[metal] = _smooth_for_plot(
        df_pH_plot[metal].to_numpy()
    )


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig1, ax1 = plt.subplots(

    figsize=(10, 6)

)


for metal in selected_metals:

    ax1.plot(

        df_pH_plot["pH"],

        df_pH_plot[metal],

        label=metal,

        linewidth=2.5

    )


ax1.set_title(

    f"Competitive {precipitant_name} Precipitation"

    +

    (

        f" with {complexant_name}"

        if use_complexant

        else ""

    )

)


ax1.set_xlabel(
    "pH"
)


ax1.set_ylabel(
    "Total Metal Precipitated (%)"
)


ax1.set_xlim(

    pH_min,

    pH_max

)


ax1.set_ylim(

    -2,

    102

)


ax1.set_xticks(

    np.arange(

        np.ceil(pH_min),

        np.floor(pH_max) + 1,

        1

    )

)


ax1.grid(

    True,

    linestyle="--",

    alpha=0.5

)


ax1.legend(
    loc="best"
)


fig1.tight_layout()


st.pyplot(

    fig1,

    use_container_width=True

)


# ============================================================
# GRAPH 2 — PRECIPITATION VS TOTAL PRECIPITANT
# ============================================================

st.header(
    "2. Precipitation as a Function of Total Precipitant"
)


st.caption(

    f"Fixed pH = {fixed_pH:.1f}"

)


# ------------------------------------------------------------
# CONCENTRATION RANGE
# ------------------------------------------------------------

max_precipitant = st.number_input(

    "Maximum total precipitant concentration "
    "for the second graph",

    min_value=0.000001,

    value=max(

        1.0,

        precipitant_total * 2

    ),

    step=0.1,

    format="%.6f"

)


concentration_values = np.linspace(

    0.0,

    max_precipitant,

    concentration_points

)


concentration_results = []


progress_bar = st.progress(0)


for i, precipitant_conc in enumerate(

    concentration_values

):

    equilibrium = solve_equilibrium(

        pH=fixed_pH,

        metals=metals,

        complexant_total=
            complexant_total,

        precipitant_total=
            precipitant_conc,

        complexant_pkas=
            complexant_pkas,

        precipitant_pkas=
            precipitant_pkas

    )


    row = {

        "Total precipitant":
            precipitant_conc

    }


    for metal in selected_metals:

        row[metal] = equilibrium.get(

            metal,

            0.0

        )


    concentration_results.append(
        row
    )


    progress_bar.progress(

        int(

            (i + 1)
            / len(concentration_values)
            * 100

        )

    )


progress_bar.empty()


df_concentration = pd.DataFrame(

    concentration_results

)


# ------------------------------------------------------------
# VISUAL SMOOTHING ONLY
# ------------------------------------------------------------

df_concentration_plot = (
    df_concentration.copy()
)

for metal in selected_metals:

    df_concentration_plot[metal] = _smooth_for_plot(
        df_concentration_plot[metal].to_numpy()
    )


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig2, ax2 = plt.subplots(

    figsize=(10, 6)

)


for metal in selected_metals:

    ax2.plot(

        df_concentration_plot[
            "Total precipitant"
        ],

        df_concentration_plot[metal],

        label=metal,

        linewidth=2.5

    )


ax2.set_title(

    f"{precipitant_name} Precipitation "
    f"at Fixed pH = {fixed_pH:.1f}"

)


ax2.set_xlabel(

    "Total Analytical Precipitant Concentration"

)


ax2.set_ylabel(

    "Total Metal Precipitated (%)"

)


ax2.set_xlim(

    0,

    max_precipitant

)


ax2.set_ylim(

    -2,

    102

)


ax2.grid(

    True,

    linestyle="--",

    alpha=0.5

)


ax2.legend(
    loc="best"
)


fig2.tight_layout()


st.pyplot(

    fig2,

    use_container_width=True

)


# ============================================================
# RESULTS TABLE
# ============================================================

with st.expander(
    "View numerical results"
):

    st.subheader(
        "pH sweep"
    )


    st.caption(
        "These are the raw numerical equilibrium results. "
        "No graphical smoothing is applied to this table."
    )


    st.dataframe(

        df_pH,

        use_container_width=True,

        hide_index=True

    )


    st.subheader(

        "Precipitant concentration sweep"

    )


    st.caption(
        "These are the raw numerical equilibrium results. "
        "No graphical smoothing is applied to this table."
    )


    st.dataframe(

        df_concentration,

        use_container_width=True,

        hide_index=True

    )
