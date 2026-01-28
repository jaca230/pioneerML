import polars as pl

from pioneerml.data.processing import assign_time_group_labels, add_time_group_labels


def _legacy_group(times, window):
    """Reference implementation matching deprecated/omar_pioneerML grouping."""
    hits = sorted(times)
    groups = []
    current = []
    last = None
    for t in hits:
        if not current:
            current.append(t)
            last = t
        elif abs(t - last) <= window:
            current.append(t)
            last = t
        else:
            groups.append(current)
            current = [t]
            last = t
    if current:
        groups.append(current)

    # Map back to original order (first matching group assignment)
    labels = []
    for t in times:
        for gid, grp in enumerate(groups):
            if grp and t in grp:
                labels.append(gid)
                grp.remove(t)  # avoid double use for duplicate times
                break
    return labels


def test_assign_time_group_labels_matches_legacy():
    times = [2.0, 0.0, 0.8, 2.2, 5.5]
    window = 1.0
    assert assign_time_group_labels(times, window) == _legacy_group(times, window)


def test_add_time_group_labels_polars():
    df = pl.DataFrame({"hits_time": [[0.0, 0.5, 2.0], [1.0, 3.5]]})
    out = add_time_group_labels(df, window_ns=1.0)
    groups = out["hits_time_group"].to_list()
    assert groups[0] == [0, 0, 1]
    assert groups[1] == [0, 1]
