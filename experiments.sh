#!/bin/sh

ENV="cartpole" # or gridworld
OUTPUT_DIR="output/$ENV"
mkdir -p "$OUTPUT_DIR"

# === ICL (change parameters if necessary)  ===
python3 -B icl.py -env "$ENV" -o "$OUTPUT_DIR/icl_0" -seed 0

# === Prob ICL (change parameters if necessary) ===
python3 -B prob_icl.py -env "$ENV" -o "$OUTPUT_DIR/ipcl0.5_0" -seed 0 -delta 0.50 -expert_dir "$OUTPUT_DIR/icl_0" # remove -expert_dir if needed

