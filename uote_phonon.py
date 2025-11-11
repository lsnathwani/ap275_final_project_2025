#!/usr/bin/env python3
# uote_phonon.py — minimal DFPT flow for UOTe with automatic D3 Hessian

import os, shutil, subprocess, textwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- Paths / executables ---
BASE = Path("/shared/home/lun364/ap275_final_project_2025").resolve()
WORK = BASE / "UOTe_phonon"
TMP  = WORK / "tmp"
PSEUDO_DIR = BASE

PW  = shutil.which("pw.x")
PH  = shutil.which("ph.x")
Q2R = shutil.which("q2r.x")
MAT = shutil.which("matdyn.x")
MPI = shutil.which("mpirun") or shutil.which("mpiexec")
NPROCS = int(os.environ.get("NPROCS", "1"))

if not all([PW, PH, Q2R, MAT]):
    raise RuntimeError("Missing QE binaries in PATH (pw.x/ph.x/q2r.x/matdyn.x)")

def run(cmd, cwd=None):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def mpicmd(exe, *args):
    return ([MPI, "-np", str(NPROCS), exe, *args] if MPI else [exe, *args])

# --- Model / inputs ---
PREFIX   = "UOTe"
VDW_CORR = "grimme-d3"   # set to "none" to skip D3/Hessian
ECUTWFC, ECUTRHO = 35, 35*6
K_AUTOMATIC = (4, 4, 2, 0, 0, 0)
TR2_PH = "1.0d-7"
NQ = (1, 1, 1)

cell = (4.050000, 4.050000, 8.650000)  # a,b,c in Å

PSEUDOS = {
    "U":  "U.paw.z_14.ld1.uni-marburg.v0.upf",
    "O":  "O.pbe-n-kjpaw_psl.0.1.UPF",
    "Te": "Te.pbe-n-kjpaw_psl.1.0.0.UPF",
}

POS = [
    ("U",  0.750000, 0.750000, 0.169414),
    ("U",  0.250000, 0.250000, 0.830586),
    ("U",  0.250000, 0.250000, 0.330586),
    ("U",  0.750000, 0.750000, 0.669414),
    ("Te", 0.750000, 0.750000, 0.927105),
    ("Te", 0.750000, 0.750000, 0.427105),
    ("Te", 0.250000, 0.250000, 0.072895),
    ("Te", 0.250000, 0.250000, 0.572895),
    ("O",  0.750000, 0.250000, 0.750000),
    ("O",  0.750000, 0.250000, 0.250000),
    ("O",  0.250000, 0.750000, 0.750000),
    ("O",  0.250000, 0.750000, 0.250000),
]

# --- Writers ---
def ensure_setup():
    WORK.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    for el, f in PSEUDOS.items():
        if not (PSEUDO_DIR / f).exists():
            raise FileNotFoundError(f"Missing pseudo for {el}: {PSEUDO_DIR/f}")

def write_pw_scf():
    species = f"""ATOMIC_SPECIES
  U  238.02891  {PSEUDOS['U']}
  O   15.999    {PSEUDOS['O']}
  Te 127.60     {PSEUDOS['Te']}"""
    cellblk = f"""CELL_PARAMETERS angstrom
  {cell[0]:.6f}  0.000000  0.000000
  0.000000  {cell[1]:.6f}  0.000000
  0.000000  0.000000  {cell[2]:.6f}"""
    pos = "\n".join(f"  {s} {x:.6f} {y:.6f} {z:.6f}" for s,x,y,z in POS)
    txt = textwrap.dedent(f"""\
        &control
          calculation='scf'
          prefix='{PREFIX}'
          pseudo_dir='{PSEUDO_DIR}'
          outdir='{TMP}/'
          verbosity='high'
          wf_collect=.true.
        /
        &system
          ibrav=0
          nat={len(POS)}
          ntyp=3
          ecutwfc={ECUTWFC}
          ecutrho={ECUTRHO}
          vdw_corr='{VDW_CORR}'
          occupations='smearing'
          smearing='mp'
          degauss=0.02
          dftd3_threebody = .false.

        /
        &electrons
          mixing_beta=0.7
          conv_thr=1.0e-6
          diagonalization='david'
        /
        {species}
        {cellblk}
        ATOMIC_POSITIONS crystal
{pos}
        K_POINTS automatic
          {K_AUTOMATIC[0]} {K_AUTOMATIC[1]} {K_AUTOMATIC[2]} {K_AUTOMATIC[3]} {K_AUTOMATIC[4]} {K_AUTOMATIC[5]}
    """)
    (WORK/"pw.scf.in").write_text(txt)

def write_ph_in():
    txt = textwrap.dedent(f"""\
        &inputph
          prefix='{PREFIX}'
          outdir='{TMP}/'
          tr2_ph={TR2_PH}
          ldisp=.true.
          nq1={NQ[0]}, nq2={NQ[1]}, nq3={NQ[2]}
          fildyn='{PREFIX}.dyn'
          epsil=.false.
        /
    """)
    (WORK/"ph.in").write_text(txt)

def write_q2r_in():
    (WORK/"q2r.in").write_text(textwrap.dedent(f"""\
        &input
          fildyn='{PREFIX}.dyn'
          flfrc='{PREFIX}.fc'
          zasr='crystal'
        /
    """))

def write_matdyn_in():
    # simple Γ–X–Γ–Y–Γ–Z with 6 points per segment
    path = [((0,0,0),6), ((1,0,0),6), ((0,0,0),6), ((0,1,0),6), ((0,0,0),6), ((0,0,1),6)]
    lines = [str(len(path))]
    for (qx,qy,qz), n in path:
        lines.append(f"  {qx:.6f} {qy:.6f} {qz:.6f}  {n}")
    body = "\n".join(lines)  # <-- build outside the f-string

    txt = textwrap.dedent(f"""\
        &input
          flfrc='{PREFIX}.fc'
          flfrq='{PREFIX}.freq'
          q_in_band_form=.true.
          q_in_cryst_coord=.true.
          asr='crystal'
        /
        {body}
    """)
    (WORK / "matdyn.in").write_text(txt)

# --- D3 Hessian (short + robust) ---
def ensure_d3_hessian():
    """Create {PREFIX}.hess in TMP if using grimme-d3."""
    if VDW_CORR.lower() == "none":
        print("vdw_corr=none: no Hessian needed.")
        return
    target = TMP / f"{PREFIX}.hess"
    if target.exists():
        print(f"Found Hessian: {target}")
        return

    d3hess = shutil.which("d3hess.x")
    if not d3hess:
        raise RuntimeError(
            "vdw_corr='grimme-d3' but d3hess.x not found.\n"
            "Install d3hess.x (QE) or disable D3."
        )

    # try stdin style first, then fallback to bare run (some builds read d3hess.in by default)
    (WORK/"d3hess.in").write_text(textwrap.dedent(f"""\
        &INPUT
          prefix='{PREFIX}'
          outdir='{TMP}/'
        /
    """))
    try:
        run(["/bin/bash", "-lc", f"cd '{WORK}'; {d3hess} < d3hess.in > d3hess.out"])
    except subprocess.CalledProcessError:
        run([d3hess], cwd=WORK)

    if not target.exists():
        raise RuntimeError(f"D3 Hessian was not produced at {target}. Check d3hess.out.")

# --- Steps ---
def scf():  run(mpicmd(PW, "-i", "pw.scf.in"), cwd=WORK)
def ph():   run(mpicmd(PH, "-i", "ph.in"), cwd=WORK)
def q2r():  run([Q2R, "-i", "q2r.in"], cwd=WORK)
def matdyn(): run([MAT, "-i", "matdyn.in"], cwd=WORK)
# put near the top (before pyplot import) if not already:
import re

def _sanitize_gp_line(line: str) -> str:
    # Insert a space before a '-' that immediately follows a digit and is NOT part of an exponent
    # e.g. "... 123.45-67.8 ..." -> "... 123.45 -67.8 ..."
    # also handle "+", just in case
    line = re.sub(r'(?<=\d)-', ' -', line)
    line = re.sub(r'(?<=\d)\+', ' +', line)
    # don't touch "E-03" etc. (handled because lookbehind is digit, not [eE])
    return line

def _read_freq_gp(path):
    xs = []
    ys = []
    with open(path, 'r') as f:
        for raw in f:
            if not raw.strip() or raw.lstrip().startswith(('#','!')):
                continue
            s = _sanitize_gp_line(raw)
            parts = s.split()
            # first column = curvilinear distance; rest = mode frequencies (cm^-1)
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                # last fallback: try splitting on multiple minus signs
                s2 = re.sub(r'([0-9])(-)', r'\1 \2', raw)
                parts = s2.split()
                vals = [float(p) for p in parts]
            xs.append(vals[0])
            ys.append(vals[1:])
    import numpy as np
    x = np.array(xs, float)
    y = np.array(ys, float)  # shape: (npoints, nmodes)
    return x, y

def plot():
    gp = WORK / f"{PREFIX}.freq.gp"
    if not gp.exists():
        raise FileNotFoundError(f"Expected {gp}; run matdyn first.")
    s, bands = _read_freq_gp(gp)

    import matplotlib
    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4))
    for i in range(bands.shape[1]):
        plt.plot(s, bands[:, i], lw=1)

    # draw segment joints if you used 6 points/segment in matdyn.in
    # (adjust if you changed the band path sampling)
    seg = 6
    joints = [i*seg-1 for i in range(1, int(len(s)/seg))]
    for j in joints:
        plt.axvline(s[j], lw=0.5)

    plt.ylabel("Frequency (cm$^{-1}$)")
    plt.title("UOTe phonon dispersion")
    plt.ylim(bottom=min(-50, float(bands.min())))
    plt.tight_layout()
    outpng = WORK / "UOTe_phonon_dispersion.png"
    plt.savefig(outpng, dpi=300)
    print(f"Saved {outpng}")

# --- Main ---
if __name__ == "__main__":
    ensure_setup()
    write_ph_in()
    write_q2r_in()
    write_matdyn_in()

    # scf()                 # <-- skip
    # ensure_d3_hessian()   # <-- skip (you already have UOTe.hess)
    ph()
    q2r()
    matdyn()
    plot()

