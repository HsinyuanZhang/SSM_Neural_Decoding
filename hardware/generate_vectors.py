"""Generate deterministic randomized Python-reference vectors for the RTL testbench."""
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ssm_decode.fixedpoint import Real2x2FixedPoint

OUT = ROOT / "hardware" / "fixedpoint_vectors.txt"

def main() -> None:
    rng = random.Random(20261003)
    # Radius ~0.96; b injects u into q. Includes values that exercise saturation.
    mode = Real2x2FixedPoint((15565, -2344, 2344, 15565), (0, 16384), state_bits=16)
    # Plain integer rows keep the synthesizer-independent $fscanf testbench simple.
    rows = []
    for index in range(256):
        u = [0, 1, -1, 8192, -8192, 30000, -30000][index] if index < 7 else rng.randint(-30000, 30000)
        p, q = mode.step(u)
        rows.append(f"{u} {p} {q} {mode.saturation_count}\n")
    OUT.write_text("".join(rows))

if __name__ == "__main__":
    main()
