#!/usr/bin/env python3
import subprocess
import statistics
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import re

def run_single_sim_per_sta(cheater_cw_dict, rng_run, verbose=False):
    """
    Run one ns-3 simulation where ONLY cheater STAs get explicit CWmin values.
    Honest STAs are left to ns-3 defaults.

    cheater_cw_dict: {sta_index: cwmin}, e.g. {0: 3, 2: 7}
    rng_run: integer passed to --RngRun
    Returns: dict {sta_index: throughput_Mbps} for all STAs seen in FlowMonitor.
    """
    # Build --cheaters string like "0:3,2:7"
    if cheater_cw_dict:
        cheaters_config = ",".join(f"{idx}:{cw}" for idx, cw in cheater_cw_dict.items())
    else:
        cheaters_config = ""

    cmd = [
        "./ns3",
        "run",
        f"selfish-cw --enableCheater=1 --cheaters={cheaters_config} --RngRun={rng_run}"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = result.stdout
    if verbose:
        print(stdout)

    # Parse per-STA throughput from lines like:
    #   FlowId 1  STA 0  src=10.1.1.2  dst=10.1.1.1  throughput=0.2310 Mbps
    pattern = re.compile(r"STA\s+(\d+).*throughput=([0-9.]+)\s+Mbps")

    sta_throughputs = {}
    for line in stdout.splitlines():
        m = pattern.search(line)
        if m:
            sta = int(m.group(1))
            thr = float(m.group(2))
            sta_throughputs[sta] = thr

    return sta_throughputs


def run_single_sim(cheaters_config, rng_run, verbose=False):
    """
    Run one ns-3 simulation for a given cheaters_config string and RngRun.

    Expects C++ to print a line like:
      RESULT,cheaters=0:3,rng=1,cheaterAvg=1.23,honestAvg=0.04

    cheaters_config: string like "0:3" or "0:3,5:3,7:3"
    Returns: (cheaterAvg, honestAvg)
    """
    cmd = [
        "./ns3",
        "run",
        f"selfish-cw --enableCheater=1 --cheaters={cheaters_config} --RngRun={rng_run}"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    if verbose:
        print(stdout)

    # Find the RESULT line
    result_line = None
    for line in stdout.splitlines():
        if line.startswith("RESULT,"):
            result_line = line
            break

    if result_line is None:
        raise RuntimeError("No RESULT line found in ns-3 output")

    # Example:
    # RESULT,cheaters=0:3,rng=1,cheaterAvg=1.23,honestAvg=0.04
    payload = result_line[len("RESULT,"):]
    parts = payload.split(",")

    kv = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip()

    try:
        cheater_avg = float(kv["cheaterAvg"])
        honest_avg = float(kv["honestAvg"])
    except KeyError as e:
        raise RuntimeError(f"Missing expected field in RESULT line: {e}, line={result_line}")

    return cheater_avg, honest_avg


def experiment1():
    """
    Experiment 1:
    - 20 STAs, 1 cheater (STA 0)
    - Sweep cheater CWmin from 1 to 30
    - For each CWmin, run 3 different RngRun values
    - Average cheater throughput and honest throughput across runs
    - Save results and plot CWmin vs (cheater throughput, avg honest throughput)
    """
    cwmin_values = list(range(1, 31))  # 1..30 inclusive
    runs_per_cw = 3

    avg_cheater_thrs = []
    avg_honest_thrs = []

    for cw in cwmin_values:
        cheater_samples = []
        honest_samples = []

        print(f"[Experiment 1] Running CWmin={cw} ...")

        cheaters_config = f"0:{cw}"  # one cheater: STA 0

        for rng_run in range(1, runs_per_cw + 1):
            cheater_thr, honest_thr = run_single_sim(cheaters_config, rng_run, verbose=False)
            cheater_samples.append(cheater_thr)
            honest_samples.append(honest_thr)

        avg_cheater = statistics.mean(cheater_samples)
        avg_honest = statistics.mean(honest_samples)

        avg_cheater_thrs.append(avg_cheater)
        avg_honest_thrs.append(avg_honest)

        print(f"  CWmin={cw}: cheater avg={avg_cheater:.4f} Mbps, "
              f"honest avg={avg_honest:.4f} Mbps over {runs_per_cw} runs")

    # ---------- SAVE RESULTS FOR LATER USE ----------
    csv_path = "game_simulations/experiment1_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cwmin", "cheater_avg_mbps", "honest_avg_mbps"])
        for cw, c_thr, h_thr in zip(cwmin_values, avg_cheater_thrs, avg_honest_thrs):
            writer.writerow([cw, c_thr, h_thr])
    print(f"[Experiment 1] Saved CSV results to {csv_path}")

    npz_path = "game_simulations/experiment1_results.npz"
    np.savez(
        npz_path,
        cwmin=np.array(cwmin_values, dtype=np.int32),
        cheater_avg=np.array(avg_cheater_thrs, dtype=np.float64),
        honest_avg=np.array(avg_honest_thrs, dtype=np.float64),
    )
    print(f"[Experiment 1] Saved NPZ results to {npz_path}")

    # ---------- PLOTTING ----------
    plt.figure(figsize=(8, 5))
    plt.plot(cwmin_values, avg_cheater_thrs, marker='o', label="Cheater throughput")
    plt.plot(cwmin_values, avg_honest_thrs, marker='s', label="Avg honest throughput")

    plt.xlabel("Cheater CWmin")
    plt.ylabel("Throughput (Mbps)")
    plt.title("Experiment 1: One cheater among 20 STAs")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig("game_simulations/experiment1_cwmin_vs_throughput.png", dpi=200)
    print("[Experiment 1] Saved plot to experiment1_cwmin_vs_throughput.png")
    plt.show()


def experiment2():
    """
    Experiment 2:
    - 20 STAs, 10 cheaters (indices chosen randomly per CWmin)
    - All cheaters use the same CWmin for a given run
    - Sweep cheater CWmin as:
        1..10 (step 1),
        15..30 (step 5),
        40..100 (step 10)
    - For each CWmin, run 3 different RngRun values
    - Average cheater throughput and honest throughput across runs
    - Save results and plot
    """
    # CWmin grid:
    cwmin_values = []
    cwmin_values.extend(range(1, 11))        # 1..10
    cwmin_values.extend(range(15, 31, 5))    # 15,20,25,30
    cwmin_values.extend(range(40, 101, 10))  # 40,50,...,100

    runs_per_cw = 5
    n_stations = 20
    n_cheaters = 10

    avg_cheater_thrs = []
    avg_honest_thrs = []
    chosen_cheater_sets = []  # to remember which indices were used per CWmin

    # Fix Python RNG seed for reproducibility of which STAs become cheaters
    random.seed(12345)

    for cw in cwmin_values:
        cheater_samples = []
        honest_samples = []

        # Randomly choose 10 distinct cheater indices out of 20
        cheater_indices = sorted(random.sample(range(n_stations), n_cheaters))
        chosen_cheater_sets.append(cheater_indices)

        cheaters_config = ",".join(f"{idx}:{cw}" for idx in cheater_indices)

        print(f"[Experiment 2] CWmin={cw}, cheaters={cheater_indices}")

        for rng_run in range(1, runs_per_cw + 1):
            cheater_thr, honest_thr = run_single_sim(cheaters_config, rng_run, verbose=False)
            cheater_samples.append(cheater_thr)
            honest_samples.append(honest_thr)

        avg_cheater = statistics.mean(cheater_samples)
        avg_honest = statistics.mean(honest_samples)

        avg_cheater_thrs.append(avg_cheater)
        avg_honest_thrs.append(avg_honest)

        print(f"  CWmin={cw}: cheater avg={avg_cheater:.4f} Mbps, "
              f"honest avg={avg_honest:.4f} Mbps over {runs_per_cw} runs")

    # ---------- SAVE RESULTS ----------
    csv_path = "game_simulations/experiment2_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cwmin", "cheater_indices", "cheater_avg_mbps", "honest_avg_mbps"])
        for cw, indices, c_thr, h_thr in zip(cwmin_values, chosen_cheater_sets,
                                             avg_cheater_thrs, avg_honest_thrs):
            idx_str = " ".join(str(i) for i in indices)
            writer.writerow([cw, idx_str, c_thr, h_thr])
    print(f"[Experiment 2] Saved CSV results to {csv_path}")

    npz_path = "game_simulations/experiment2_results.npz"
    np.savez(
        npz_path,
        cwmin=np.array(cwmin_values, dtype=np.int32),
        cheater_avg=np.array(avg_cheater_thrs, dtype=np.float64),
        honest_avg=np.array(avg_honest_thrs, dtype=np.float64),
        cheater_indices=np.array(chosen_cheater_sets, dtype=object),
    )
    print(f"[Experiment 2] Saved NPZ results to {npz_path}")

    # ---------- PLOTTING ----------
    plt.figure(figsize=(8, 5))
    plt.plot(cwmin_values, avg_cheater_thrs, marker='o', label="Cheaters' avg throughput")
    plt.plot(cwmin_values, avg_honest_thrs, marker='s', label="Avg honest throughput")

    plt.xlabel("Cheater CWmin")
    plt.ylabel("Throughput (Mbps)")
    plt.title("Experiment 2: 10 cheaters among 20 STAs")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig("game_simulations/experiment2_cwmin_vs_throughput.png", dpi=200)
    print("[Experiment 2] Saved plot to experiment2_cwmin_vs_throughput.png")
    plt.show()


def experiment3_bandit_punishment():
    """
    Experiment 3:
    Bandit-style CW learning for cheater STAs only + punishment.

    - 20 STAs total.
    - cheater_ids: bandit players (we only change their CW).
    - Honest STAs: ns-3 default CW, never touched by Python.
    - Each cheater runs epsilon-greedy over CW_ARMS.
    - Punishment: if cheater's throughput > (1+DELTA)*fair_share,
      its CW is forced to CW_PUNISH for PUNISH_ROUNDS epochs.
    """

    n_stations = 20
    cheater_ids = [0, 1, 2, 6, 7, 11, 13, 14, 18, 19]                   # GAME PLAYERS
    honest_ids = [i for i in range(n_stations) if i not in cheater_ids]

    # Bandit CW arms
    CW_ARMS = [1, 2, 3, 5, 7, 15, 31, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    K = len(CW_ARMS)

    # Punishment params (for cheaters only)
    CW_PUNISH = 1023
    DELTA = 0.20          # 20% above fair share threshold
    PUNISH_ROUNDS = 1

    # Bandit params
    EPSILON_INIT = 0.5     # starting exploration rate
    EPSILON_MIN  = 0.005    # don't go below this
    EPSILON_DECAY = 0.9   # per-round multiplicative decay
    BANDIT_ALPHA = 0.5

    # Simulation control
    NUM_ROUNDS = 50
    runs_per_round = 1    # could increase if you want intra-epoch averaging

    # ---- Initialize bandit state for each cheater ----
    bandit_state = {}
    for cid in cheater_ids:
        bandit_state[cid] = {
            "counts": [0] * K,
            "values": [0.0] * K,   # estimated mean reward per CW arm
            "last_arm": None,
        }

    # Punishment counters for cheaters only
    punish_counter = {cid: 0 for cid in cheater_ids}

    # Logs
    rounds = []
    avg_cheater_thrs_over_rounds = []
    avg_honest_thrs_over_rounds = []
    avg_cheater_cw_over_rounds = []

    random.seed(2025)

    for round_idx in range(NUM_ROUNDS):
        print(f"[Experiment 3] Round {round_idx}/{NUM_ROUNDS - 1}")

        EPSILON = max(EPSILON_MIN, EPSILON_INIT * (EPSILON_DECAY ** round_idx))
        print(f"[Experiment 3] Round {round_idx}/{NUM_ROUNDS - 1}, epsilon={EPSILON:.3f}")

        # 1) Each cheater picks a CW arm via epsilon-greedy
        chosen_arm_for_cheater = {}
        cheater_cw_dict = {}

        for cid in cheater_ids:
            state = bandit_state[cid]
            counts = state["counts"]
            values = state["values"]

            # ε-greedy: explore or exploit
            if random.random() < EPSILON:
                arm = random.randrange(K)
            else:
                max_value = max(values)
                best_arms = [i for i, v in enumerate(values) if v == max_value]
                arm = random.choice(best_arms)

            state["last_arm"] = arm
            chosen_arm_for_cheater[cid] = arm

            # Apply punishment override if active
            if punish_counter[cid] > 0:
                cheater_cw_dict[cid] = CW_ARMS[arm] #CW_PUNISH
                values[arm] = values[arm] - 5  # punishment on value function
            else:
                cheater_cw_dict[cid] = CW_ARMS[arm]

        print(f"  Cheater CWs this round: {cheater_cw_dict}")

        # 2) Run ns-3 with these cheater CWs (honest STAs untouched)
        cheater_thrs = []
        honest_thrs = []

        for run in range(runs_per_round):
            rng_run = round_idx * runs_per_round + run + 1
            sta_throughputs = run_single_sim_per_sta(cheater_cw_dict, rng_run, verbose=False)

            for i, thr in sta_throughputs.items():
                if i in cheater_ids:
                    cheater_thrs.append(thr)
                elif i in honest_ids:
                    honest_thrs.append(thr)

        avg_cheater_thr = max(cheater_thrs) if cheater_thrs else 0.0
        avg_honest_thr = statistics.mean(honest_thrs) if honest_thrs else 0.0

        print(f"  Max cheater throughput: {avg_cheater_thr:.4f} Mbps")
        print(f"  Avg honest throughput:  {avg_honest_thr:.4f} Mbps")

        rounds.append(round_idx)
        avg_cheater_thrs_over_rounds.append(avg_cheater_thr)
        avg_honest_thrs_over_rounds.append(avg_honest_thr)

        # For plotting: average CW that cheaters *intended* to use this round
        mean_cw_cheaters = statistics.mean(
            [CW_ARMS[chosen_arm_for_cheater[cid]] for cid in cheater_ids]
        )
        avg_cheater_cw_over_rounds.append(mean_cw_cheaters)

        # 3) Bandit updates: use each cheater's throughput as reward
        last_sta_throughputs = sta_throughputs  # from last ns-3 run

        for cid in cheater_ids:
            reward = last_sta_throughputs.get(cid, 0.0)
            arm = chosen_arm_for_cheater[cid]
            state = bandit_state[cid]
            counts = state["counts"]
            values = state["values"]

            counts[arm] += 1
            n = counts[arm]
            old_value = values[arm]
            new_value = (1.0 - BANDIT_ALPHA) * old_value + BANDIT_ALPHA * reward
            values[arm] = new_value

            print(f"    Cheater {cid}: arm={arm} (CW={CW_ARMS[arm]}), "
                  f"reward={reward:.4f}, new_est={new_value:.4f}")

        # 4) Punishment: only cheaters can be punished
        total_thr = sum(last_sta_throughputs.values())
        fair_share = total_thr / n_stations if n_stations > 0 else 0.0
        threshold = (1.0 + DELTA) * fair_share

        print(f"  Total thr={total_thr:.4f} Mbps, fair_share≈{fair_share:.4f}, "
              f"punish_threshold≈{threshold:.4f}")

        for cid in cheater_ids:
            thr_i = last_sta_throughputs.get(cid, 0.0)

            # Decrement existing punishment
            if punish_counter[cid] > 0:
                punish_counter[cid] -= 1

            # Apply / refresh punishment if cheater is too dominant
            if thr_i > threshold:
                punish_counter[cid] = PUNISH_ROUNDS
                print(f"    Cheater STA {cid} flagged as selfish: thr={thr_i:.4f}, "
                      f"punish for {PUNISH_ROUNDS} rounds")

        print("  Punish counters (cheaters only):", punish_counter)
        print("")

    # ----- SAVE / PLOT DYNAMICS -----

    csv_path = "game_simulations/experiment3_bandit_dynamics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "avg_cheater_mbps", "avg_honest_mbps", "avg_cheater_cw"])
        for r, c_thr, h_thr, cw_bar in zip(
            rounds,
            avg_cheater_thrs_over_rounds,
            avg_honest_thrs_over_rounds,
            avg_cheater_cw_over_rounds
        ):
            writer.writerow([r, c_thr, h_thr, cw_bar])
    print("[Experiment 3] Saved CSV to experiment3_bandit_dynamics.csv")

    npz_path = "game_simulations/experiment3_bandit_dynamics.npz"
    np.savez(
        npz_path,
        rounds=np.array(rounds, dtype=np.int32),
        avg_cheater=np.array(avg_cheater_thrs_over_rounds, dtype=np.float64),
        avg_honest=np.array(avg_honest_thrs_over_rounds, dtype=np.float64),
        avg_cheater_cw=np.array(avg_cheater_cw_over_rounds, dtype=np.float64),
    )
    print("[Experiment 3] Saved NPZ to experiment3_bandit_dynamics.npz")

    # Throughput evolution
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, avg_cheater_thrs_over_rounds,
             marker='o', label="Cheaters' max throughput")
    plt.plot(rounds, avg_honest_thrs_over_rounds,
             marker='s', label="Honest avg throughput")
    plt.xlabel("Round")
    plt.ylabel("Throughput (Mbps)")
    plt.title("Experiment 3: Bandit CW (cheaters only) + Punishment")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("game_simulations/experiment3_bandit_throughput.png", dpi=200)
    print("[Experiment 3] Saved plot to experiment3_bandit_throughput.png")
    plt.show()

    # CW choices over time
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, avg_cheater_cw_over_rounds,
             marker='^', label="Mean CW chosen by cheaters")
    plt.xlabel("Round")
    plt.ylabel("CWmin")
    plt.title("Experiment 3: Bandit CW choices over time (cheaters only)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("game_simulations/experiment3_bandit_cw_choices.png", dpi=200)
    print("[Experiment 3] Saved plot to experiment3_bandit_cw_choices.png")
    plt.show()


if __name__ == "__main__":
    # Uncomment what you want to run
    # experiment1()
    # experiment2()
    experiment3_bandit_punishment()
