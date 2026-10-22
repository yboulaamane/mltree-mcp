"""Shared helpers (logging, seeding)."""
import random


def seed_everything(seed=42):
    random.seed(seed)


def log(msg):
    print(f"[proj] {msg}")
