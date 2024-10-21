import argparse
import pandas as pd
import numpy as np
import json
from scipy import stats

from math import comb
from tqdm import tqdm
from glob import glob
from collections import defaultdict

import plotly.express as px
import requests

from itertools import combinations

pd.options.display.float_format = '{:.2f}'.format

def _get_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def _get_interval(df, model):
    row = df[df.model == model]
    assert len(row) == 1
    return (row.iloc[0].lower, row.iloc[0].upper)

def _get_elo_interval(df, model):
    row = df[df["index"] == model]
    assert len(row) == 1
    return (row.iloc[0].rating_q025, row.iloc[0].rating_q975)

def get_unique_overlapping_interval_pairs(df: pd.DataFrame, key_lower: str ="lower", key_upper: str ="upper"):
    intervals = [[lower, upper] for lower, upper in zip(df[key_lower].tolist(), df[key_upper].tolist())]

    # Sort the intervals by start time
    intervals.sort(key=lambda x: x[0])

    overlapping_pairs = []
    for i in range(len(intervals)):
        for j in range(i+1, len(intervals)):
            # If the start time of the second interval is less than the end time of the first, they overlap
            if intervals[j][0] < intervals[i][1]:
                # Check if the pair is already in the list
                if (intervals[i], intervals[j]) not in overlapping_pairs and (intervals[j], intervals[i]) not in overlapping_pairs:
                    overlapping_pairs.append((intervals[i], intervals[j]))
            else:
                break

    num_overlapping_pairs = len(overlapping_pairs)
    total_model_pairs = comb(len(df), 2)
    return num_overlapping_pairs, total_model_pairs

def get_agreement_with_confidence(arena_hard: pd.DataFrame, elo_subset: pd.DataFrame):
    count = 0
    total = 0

    shared_models = list(set(arena_hard.model) & set(elo_subset["index"].tolist()))
    for model_a, model_b in combinations(shared_models, 2):
        score_a = _get_interval(arena_hard, model_a)
        score_b = _get_interval(arena_hard, model_b)

        elo_a = _get_elo_interval(elo_subset, model_a)
        elo_b = _get_elo_interval(elo_subset, model_b)

        if _get_overlap(elo_a, elo_b) > 0:
            continue

        total += 1

        if _get_overlap(score_a, score_b) > 0:
            count += 0.5
            continue

        if score_a < score_b and elo_a < elo_b:
            count += 1
        elif score_a > score_b and elo_a > elo_b:
            count += 1
    
    return count / total

def get_spearman_correlation(arena_hard: pd.DataFrame, elo_subset: pd.DataFrame):
    elo_subset = elo_subset.copy()
    elo_subset = elo_subset.rename(columns={"index":"model"})
    merged = pd.merge(elo_subset, arena_hard, on="model")
    res = stats.spearmanr(merged.rating, merged.score)
    return res.statistic

def get_kendall_tau_correlation(arena_hard: pd.DataFrame, elo_subset: pd.DataFrame):
    elo_subset = elo_subset.copy()
    elo_subset = elo_subset.rename(columns={"index":"model"})
    merged = pd.merge(elo_subset, arena_hard, on="model")
    res = stats.kendalltau(merged.rating, merged.score)
    return res.statistic

def print_overlap_count(arena_hard: pd.DataFrame):
    count, total = get_unique_overlapping_interval_pairs(arena_hard)
    print(f"Overlap Pair #: {count}, Total model Pair #: {total}")
    overlap_percentage = np.round(count / total * 100, decimals=2)
    print(f"Overlapped: {overlap_percentage}%")
    print(f"Confidence: {100 - overlap_percentage}%")

def print_agreement_with_confidence(arena_hard: pd.DataFrame, elo_subset: pd.DataFrame):
    agreement = get_agreement_with_confidence(arena_hard, elo_subset)
    print(f"Arena Hard Agreement with Chatbot Arena (With Confidence): {agreement}")

def print_spearman_correlation(arena_hard: pd.DataFrame, elo_subset: pd.DataFrame):
    correlation = get_spearman_correlation(arena_hard, elo_subset)
    print(f"Arena Hard (Spearman Correlation): {correlation}")

def print_kendall_tau_correlation(arena_hard: pd.DataFrame, elo_subset: pd.DataFrame):
    correlation = get_kendall_tau_correlation(arena_hard, elo_subset)
    print(f"Arena Hard (Kendall Tau Correlation): {correlation}")

def convert_arena_hard_to_arena_keys(arena_hard: pd.DataFrame, arena_to_arena_hard: dict):
    arena_hard_to_arena = {v: k for k, v in arena_to_arena_hard.items()}
    arena_hard["model"] = arena_hard.model.str.lower()
    arena_hard = arena_hard[arena_hard.model.isin(list(arena_hard_to_arena.keys()))]
    arena_hard["model"] = [arena_hard_to_arena[model] for model in arena_hard.model]
    arena_hard = arena_hard.reset_index(drop=True)
    return arena_hard

def curate_elo_subset(elo_leaderboard_file: str, arena_to_arena_hard: dict, arena_hard: pd.DataFrame, subset: str ="vision"):
    elo_results = pd.read_pickle(elo_leaderboard_file)
    if subset in elo_results:
        try:
            elo = elo_results[subset]['full']['leaderboard_table_df'].reset_index()
        except KeyError:
            elo = elo_results[subset]["leaderboard_table_df"].reset_index()
            print(elo["index"])
    else:
        elo = elo_results["full"]["leaderboard_table_df"].reset_index()

    elo_subset = elo[elo["index"].isin(arena_hard['model'])]
    elo_subset.reset_index(drop=True, inplace=True)
    return elo_subset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena_hard_file", type=str, required=True)
    parser.add_argument("--arena_to_arena_hard_file", type=str, required=True)
    parser.add_argument("--elo_leaderboard_file", type=str, required=True)
    parser.add_argument("--subset", type=str, default="vision")
    args = parser.parse_args()

    # Load Arena Hard
    arena_hard = pd.read_json(args.arena_hard_file, lines=True)
    with open(args.arena_to_arena_hard_file, "r") as f:
        arena_to_arena_hard = json.load(f)

    arena_hard = convert_arena_hard_to_arena_keys(arena_hard, arena_to_arena_hard)
    elo_subset = curate_elo_subset(args.elo_leaderboard_file, arena_to_arena_hard, arena_hard, args.subset)
    print(elo_subset)
    print(arena_hard)

    print_overlap_count(arena_hard)
    print_agreement_with_confidence(arena_hard, elo_subset)
    print_spearman_correlation(arena_hard, elo_subset)
    print_kendall_tau_correlation(arena_hard, elo_subset)