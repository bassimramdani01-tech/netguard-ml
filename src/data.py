"""Data loading for the NSL-KDD intrusion-detection dataset.

NSL-KDD is a cleaned, widely used version of the 1999 KDD Cup intrusion
dataset. Each record describes a network connection with 41 features and a
label that is either ``normal`` or one of ~20 attack types grouped into four
families: DoS, Probe, R2L (remote-to-local) and U2R (user-to-root).

The training set (KDDTrain+.txt, 125,973 rows) and test set (KDDTest+.txt,
22,544 rows) are drawn from *different* distributions: the test set contains
about 21 attack types that never appear in training, making it a realistic
generalisation benchmark.
"""
from __future__ import annotations

import os
import urllib.request
from typing import Tuple

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# The 41 features, in file order.
COLUMNS: list[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CATEGORICAL: list[str] = ["protocol_type", "service", "flag"]
NUMERIC: list[str] = [c for c in COLUMNS if c not in CATEGORICAL]

# Map every concrete attack label to one of the four attack families.
ATTACK_TO_GROUP: dict[str, str] = {}
for _a in ["back", "land", "neptune", "pod", "smurf", "teardrop",
           "apache2", "udpstorm", "processtable", "mailbomb"]:
    ATTACK_TO_GROUP[_a] = "dos"
for _a in ["satan", "ipsweep", "portsweep", "nmap", "mscan", "saint"]:
    ATTACK_TO_GROUP[_a] = "probe"
for _a in ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
           "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
           "httptunnel", "sendmail", "named", "worm"]:
    ATTACK_TO_GROUP[_a] = "r2l"
for _a in ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
           "xterm", "ps"]:
    ATTACK_TO_GROUP[_a] = "u2r"
ATTACK_TO_GROUP["normal"] = "normal"

GROUP_TO_ID: dict[str, int] = {"normal": 0, "dos": 1, "probe": 2, "r2l": 3, "u2r": 4}
ID_TO_GROUP: dict[int, str] = {v: k for k, v in GROUP_TO_ID.items()}
GROUP_LABELS: list[str] = ["normal", "DoS", "Probe", "R2L", "U2R"]


def download(force: bool = False) -> None:
    """Download KDDTrain+.txt and KDDTest+.txt into ``data/`` if missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    mirrors = [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/",
        "https://raw.githubusercontent.com/kukudux/NSL-KDD/master/",
        "https://raw.githubusercontent.com/Sandbernar/NSL-KDD/master/",
    ]
    for name in ["KDDTrain+.txt", "KDDTest+.txt"]:
        dest = os.path.join(DATA_DIR, name)
        if os.path.exists(dest) and not force:
            continue
        for base in mirrors:
            url = base + name
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                blob = urllib.request.urlopen(req, timeout=60).read()
                if len(blob) > 500_000:
                    with open(dest, "wb") as fh:
                        fh.write(blob)
                    print(f"Downloaded {name} ({len(blob):,} bytes)")
                    break
            except Exception as exc:  # try next mirror
                print(f"  mirror failed for {name}: {base} ({exc!r})")
        else:
            raise RuntimeError(f"Could not download {name}. Add it manually to data/")


def load_raw(path: str) -> pd.DataFrame:
    """Read one NSL-KDD text file into a DataFrame with named columns."""
    df = pd.read_csv(path, header=None)
    n = df.shape[1]
    if n == 42:  # training file: 41 features + label
        df.columns = COLUMNS + ["label"]
    elif n == 43:  # test file: 41 features + label + difficulty score
        df.columns = COLUMNS + ["label", "difficulty"]
    else:
        raise ValueError(f"Unexpected column count {n} in {path}")
    df["label"] = df["label"].astype(str).str.strip()
    return df


def to_targets(labels: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Map raw labels to (binary, multiclass) integer targets.

    binary:    0 = normal, 1 = attack
    multiclass: 0 = normal, 1 = DoS, 2 = Probe, 3 = R2L, 4 = U2R
    """
    groups = labels.map(ATTACK_TO_GROUP)
    unknown = labels[groups.isna()].unique()
    if len(unknown):
        raise ValueError(f"Unmapped labels found: {unknown}")
    y_multi = groups.map(GROUP_TO_ID).astype(int)
    y_bin = (groups != "normal").astype(int)
    return y_bin, y_multi


def load_train_test() -> Tuple[pd.DataFrame, pd.DataFrame,
                                pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (X_train, X_test, y_bin_train, y_multi_train, y_bin_test, y_multi_test)."""
    download()
    tr = load_raw(os.path.join(DATA_DIR, "KDDTrain+.txt"))
    te = load_raw(os.path.join(DATA_DIR, "KDDTest+.txt"))
    ytr_bin, ytr_multi = to_targets(tr["label"])
    yte_bin, yte_multi = to_targets(te["label"])
    return tr[COLUMNS], te[COLUMNS], ytr_bin, ytr_multi, yte_bin, yte_multi


if __name__ == "__main__":
    Xtr, Xte, yb_tr, ym_tr, yb_te, ym_te = load_train_test()
    print(f"Train: {Xtr.shape[0]:,} rows x {Xtr.shape[1]} features")
    print(f"Test:  {Xte.shape[0]:,} rows x {Xte.shape[1]} features")
    print("\nTrain class distribution (multiclass):")
    print(ym_tr.map(ID_TO_GROUP).value_counts())
    print("\nTest class distribution (multiclass):")
    print(ym_te.map(ID_TO_GROUP).value_counts())
