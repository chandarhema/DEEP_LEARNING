
# ============================================================
# DATA LOADING CHEAT SHEET
# File Format -> Python Library -> Loading Method
# ============================================================


# ============================================================
# 1. TABULAR DATA
# ============================================================

# -------------------------
# CSV
# -------------------------
import pandas as pd

df = pd.read_csv("data.csv")


# -------------------------
# TSV
# -------------------------
df = pd.read_csv("data.tsv", sep="\t")


# -------------------------
# PARQUET
# -------------------------
df = pd.read_parquet("data.parquet")


# -------------------------
# JSON
# -------------------------
import json

with open("data.json", "r") as f:
    data = json.load(f)

# OR using pandas
df = pd.read_json("data.json")


# -------------------------
# JSONL / JSON Lines
# -------------------------
df = pd.read_json("data.jsonl", lines=True)


# -------------------------
# ARROW
# -------------------------
import pyarrow as pa

with pa.memory_map("data.arrow", "r") as source:
    table = pa.ipc.open_file(source).read_all()

df = table.to_pandas()


# ============================================================
# 2. NUMPY DATA
# ============================================================

import numpy as np


# -------------------------
# NPY
# -------------------------
data = np.load("data.npy")

print(data.shape)
print(data.dtype)


# -------------------------
# NPZ
# -------------------------
data = np.load("data.npz")

# See available arrays
print(data.files)

# Example
# X = data["X"]
# y = data["y"]


# ============================================================
# 3. HDF5 DATA
# ============================================================

import h5py


# -------------------------
# H5
# -------------------------
data = h5py.File("data.h5", "r")

print(list(data.keys()))

# Example
# X = data["X"][:]


# -------------------------
# HDF5
# -------------------------
data = h5py.File("data.hdf5", "r")

print(list(data.keys()))


# -------------------------
# HDF5 created by pandas
# -------------------------
df = pd.read_hdf("data.h5")


# ============================================================
# 4. PYTHON PICKLE
# ============================================================

import pickle


# -------------------------
# PKL
# -------------------------
with open("data.pkl", "rb") as f:
    data = pickle.load(f)


# ============================================================
# 5. PYTORCH
# ============================================================

import torch


# -------------------------
# PT
# -------------------------
data = torch.load("data.pt")


# -------------------------
# PTH
# -------------------------
data = torch.load("data.pth")


# -------------------------
# PTH containing model state_dict
# -------------------------
# state_dict = torch.load("model.pth")
# model.load_state_dict(state_dict)


# ============================================================
# 6. TENSORFLOW
# ============================================================

import tensorflow as tf


# -------------------------
# TFRECORD
# -------------------------
dataset = tf.data.TFRecordDataset("data.tfrecord")


# -------------------------
# TFRECORDS
# -------------------------
dataset = tf.data.TFRecordDataset("data.tfrecords")


# ============================================================
# 7. BIOLOGICAL SEQUENCE DATA
# ============================================================

from Bio import SeqIO


# -------------------------
# FASTA .fa
# -------------------------
records = SeqIO.parse("sequence.fa", "fasta")

for record in records:
    print(record.id)
    print(record.seq)


# -------------------------
# FASTA .fasta
# -------------------------
records = SeqIO.parse("sequence.fasta", "fasta")

for record in records:
    print(record.id)
    print(record.seq)


# -------------------------
# FASTQ .fastq
# -------------------------
records = SeqIO.parse("reads.fastq", "fastq")

for record in records:
    print(record.id)
    print(record.seq)
    print(record.letter_annotations["phred_quality"])


# -------------------------
# FASTQ .fq
# -------------------------
records = SeqIO.parse("reads.fq", "fastq")


# ============================================================
# 8. VCF - VARIANT DATA
# ============================================================

import pysam


# -------------------------
# VCF
# -------------------------
vcf = pysam.VariantFile("variants.vcf")

for record in vcf:
    print(record.chrom)
    print(record.pos)
    print(record.ref)
    print(record.alts)


# -------------------------
# VCF using cyvcf2
# -------------------------
# from cyvcf2 import VCF
#
# vcf = VCF("variants.vcf")
#
# for variant in vcf:
#     print(variant.CHROM)
#     print(variant.POS)
#     print(variant.REF)
#     print(variant.ALT)


# ============================================================
# 9. MEDICAL IMAGING
# ============================================================

import nibabel as nib


# -------------------------
# NII
# -------------------------
img = nib.load("brain.nii")

data = img.get_fdata()

print(data.shape)


# -------------------------
# NII.GZ
# -------------------------
img = nib.load("brain.nii.gz")

data = img.get_fdata()

print(data.shape)


# ============================================================
# 10. EEG / PHYSIOLOGICAL SIGNAL DATA
# ============================================================

import mne


# -------------------------
# EDF
# -------------------------
raw = mne.io.read_raw_edf(
    "recording.edf",
    preload=True
)

data = raw.get_data()

print(data.shape)


# ============================================================
# 11. AUDIO DATA
# ============================================================

import soundfile as sf


# -------------------------
# WAV
# -------------------------
data, samplerate = sf.read("audio.wav")

print(data.shape)
print(samplerate)


# -------------------------
# FLAC
# -------------------------
data, samplerate = sf.read("audio.flac")

print(data.shape)
print(samplerate)


# ============================================================
# 12. IMAGE DATA
# ============================================================

from PIL import Image


# -------------------------
# JPG
# -------------------------
img = Image.open("image.jpg")

print(img.size)


# -------------------------
# JPEG
# -------------------------
img = Image.open("image.jpeg")

print(img.size)


# -------------------------
# PNG
# -------------------------
img = Image.open("image.png")

print(img.size)


# Convert image to NumPy array
image_array = np.array(img)

print(image_array.shape)


# ============================================================
# 13. VIDEO DATA
# ============================================================

import cv2


# -------------------------
# MP4
# -------------------------
cap = cv2.VideoCapture("video.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # frame is a NumPy array
    # Process frame here

cap.release()


# -------------------------
# AVI
# -------------------------
cap = cv2.VideoCapture("video.avi")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # frame is a NumPy array
    # Process frame here

cap.release()


# ============================================================
# QUICK REFERENCE
# ============================================================

"""
FILE FORMAT       LIBRARY

.csv              pandas
.tsv              pandas
.parquet          pandas / pyarrow
.json             json / pandas
.jsonl            pandas
.arrow            pyarrow

.npy              numpy
.npz              numpy

.h5               h5py / pandas
.hdf5             h5py / pandas

.pkl              pickle

.pt               PyTorch
.pth              PyTorch

.tfrecord         TensorFlow
.tfrecords        TensorFlow

.fa               Biopython
.fasta            Biopython
.fastq            Biopython
.fq               Biopython

.vcf              pysam / cyvcf2

.nii              NiBabel
.nii.gz           NiBabel

.edf              MNE

.wav              soundfile / scipy
.flac             soundfile

.jpg              Pillow
.jpeg             Pillow
.png              Pillow

.mp4              OpenCV
.avi              OpenCV
"""


# ============================================================
# DOMAIN-WISE MEMORY TRICK
# ============================================================

"""
TABULAR
.csv .tsv .parquet .json .jsonl
                -> pandas

NUMERICAL
.npy .npz
        -> NumPy

HDF5
.h5 .hdf5
        -> h5py

PYTORCH
.pt .pth
        -> torch

TENSORFLOW
.tfrecord .tfrecords
        -> tensorflow

GENOMICS
.fa .fasta .fastq .fq
        -> Biopython

VARIANTS
.vcf
        -> pysam / cyvcf2

MEDICAL IMAGING
.nii .nii.gz
        -> nibabel

EEG
.edf
        -> mne

AUDIO
.wav .flac
        -> soundfile

IMAGE
.jpg .jpeg .png
        -> PIL / Pillow

VIDEO
.mp4 .avi
        -> OpenCV
"""
