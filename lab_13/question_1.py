
"""
Gene accessibility prediction using a 1D CNN.

Input files expected in the same directory:
  ENCFF540FNK.bed.gz  - ENCODE K562 ATAC-seq IDR-thresholded peaks (GRCh38)
  chr9.fa             - GRCh38 chromosome 9 FASTA
  chr22.fa            - GRCh38 chromosome 22 FASTA

This coursework version uses regions on chromosomes 9 and 22 only.
It creates positive examples from accessible peaks and negative examples
from random 600-bp windows that do not overlap any listed peak on those
chromosomes. Negative labels are approximate background examples, not
experimentally confirmed inaccessible regions.
"""

import gzip
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight


# ------------------------- Configuration -------------------------
SEED = 37
SEQUENCE_LENGTH = 600
N_PER_CLASS = 2000          # 2,000 accessible + 2,000 background examples
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.0007

PEAK_FILE = "ENCFF540FNK.bed.gz"
REFERENCE_FILES = {"chr9": "chr9.fa", "chr22": "chr22.fa"}
OUTPUT_DIR = "gene_accessibility_results"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_fasta(path):
    """Read a single-chromosome FASTA file and return its sequence."""
    chunks = []
    with open(path, "r") as handle:
        for line in handle:
            if not line.startswith(">"):
                chunks.append(line.strip().upper())
    sequence = "".join(chunks)
    if not sequence:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return sequence


def read_peaks(path, allowed_chromosomes):
    """Read BED/narrowPeak intervals, retaining chr9 and chr22 only."""
    opener = gzip.open if path.endswith(".gz") else open
    peaks = {chrom: [] for chrom in allowed_chromosomes}

    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            # Also tolerate whitespace-delimited BED files.
            if len(fields) < 3:
                fields = line.split()
            if len(fields) < 3:
                continue

            chrom = fields[0]
            if chrom not in peaks:
                continue

            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                continue

            if start < 0 or end <= start:
                continue

            # narrowPeak column 10 (index 9) is summit offset from start.
            summit_offset = -1
            if len(fields) >= 10:
                try:
                    summit_offset = int(fields[9])

