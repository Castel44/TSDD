import numpy as np


def group(L):
    if len(L) == 0:
        return []
    else:
        first = last = L[0]
        for n in L[1:]:
            if n - 1 == last:  # Part of the group, bump the end
                last = n
            else:  # Not part of the group, yield current group and start a new
                yield first, last
                first = last = n
        yield first, last  # Yield the last group


def evaluate_continue_drift(true_drift_points, predicted_drift_points, total_points):
    if true_drift_points[1] == true_drift_points[0]:
        total_drifted_points = total_points - true_drift_points[0]
    else:
        total_drifted_points = true_drift_points[1] - true_drift_points[0]
    consecutive_groupus = list(group(predicted_drift_points))

    drift_segments = len(consecutive_groupus)

    false_alarm = 0
    true_positive = 0
    delay_flag = False
    delays = []
    # Check first drift
    for t in predicted_drift_points:
        if t < true_drift_points[0]:
            false_alarm += 1
        elif true_drift_points[0] <= t <= true_drift_points[1]:
            true_positive += 1
            if not delay_flag:
                delays.append(t - true_drift_points[0])
                delay_flag = True

    undetected = total_drifted_points - true_positive
    # True positive ratio
    TPR = true_positive / total_drifted_points
    # False negative ratio
    FNR = undetected / total_drifted_points
    if TPR + FNR != 1: print('Something does not add up')

    # Analyze drift to normal
    delay_flag = False
    for t in consecutive_groupus:
        if t[0] > true_drift_points[1]:
            false_alarm += t[1] - t[0]

        if t[1] > true_drift_points[1] and delay_flag is False:
            delays.append(t[1] - true_drift_points[1])
            delay_flag = True

    # False positive ratio
    FPR = false_alarm / total_points

    dict_results = {"false_alarms_points": false_alarm,
                    "drift_points_detected": true_positive,
                    "drift_points_not_detected": undetected,
                    "delays": delays,
                    "drift_segments": drift_segments,
                    "true_positive_ratio": TPR,
                    "false_negative_ratio": FNR,
                    "false_positive_ratio": FPR
                    }
    return dict_results


def evaluate_drift_sota(true_concept_drifts, pred_concept_drifts, tol=50):
    """
    Set tol same as warning window for continous drift methods.
    """
    false_alarms = 0
    drift_detected = 0
    drift_not_detected = 0
    delays = []

    # Check for false alarms
    for t in pred_concept_drifts:
        b = False
        for dt in true_concept_drifts:
            if dt <= t and t <= dt + tol:
                b = True
                break
        if b is False:  # False alarm
            false_alarms += 1

    # Check for detected and undetected drifts
    for dt in true_concept_drifts:
        b = False
        for t in pred_concept_drifts:
            if dt <= t and t <= dt + tol:
                b = True
                drift_detected += 1
                delays.append(t - dt)
                # break
        if b is False:
            drift_not_detected += 1

    N_pred_drifts = len(pred_concept_drifts)
    N_true_drifts = len(true_concept_drifts)

    TPR = min(drift_detected, N_true_drifts) / N_true_drifts
    FPR = 1 - np.divide(N_true_drifts, N_pred_drifts)
    FNR = drift_not_detected / N_true_drifts

    return {"false_alarms": false_alarms, "drift_detected": drift_detected, "drift_not_detected": drift_not_detected,
            "delays": delays, "drifts": N_pred_drifts, "true_positive_ratio": TPR,
            "false_negative_ratio": FNR, "false_positive_ratio": FPR}


def evaluate_single_drift(true_idx, pred_idx, prefix='drift'):
    false_positive = 0
    delay = -10
    for dt in pred_idx:
        if dt < true_idx:
            false_positive += 1
        else:
            delay = dt - true_idx
            break

    return {f'{prefix}_false_positive': false_positive, f'{prefix}_FPR': false_positive / true_idx,
            f'{prefix}_delay': delay}


if __name__ == "__main__":
    # Test - debug
    # Example
    drift_idx = list(range(3, 6)) + list(range(12, 16)) + list(range(18, 25)) + list(range(29, 33)) + list(
        range(40, 51))
    drift_idx_sota = [6, 12, 25, 35, 50]
    tol = 10
    t_start = 10
    t_end = 30
    true_drift = [10, 30]
    total_points = 70

    r = evaluate_continue_drift(true_drift, drift_idx, total_points)
    r2 = evaluate_drift_sota(true_drift, drift_idx_sota, tol)

    r3 = evaluate_drift_sota(true_drift, [x for y in list(group(drift_idx)) for x in y], tol)

    print(f"Continue drift: {r}\n")
    print(f"Sota drift: {r2}\n")
    print(f"Sota-like continnue drift: {r3}")
