#!/usr/bin/env python3
# gaas_phonons.py — DFPT phonon dispersion (GaAs) using your env paths

import os, shutil, subprocess, textwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------- Environment ----------
WORKDIR = Path(os.environ.get("WORKDIR", ".")).resolve() / "phonons_GaAs"
TMPDIR = WORKDIR / "tmp"
PSEUDO_DIR = Path(os.environ.get("QE_POTENTIALS",".")).resolve()
print(PSEUDO_DIR)
PW_CMD = os.environ.get("QE_PW_COMMAND") or shutil.which("pw.x")
PH_CMD = os.environ.get("QE_PH_COMMAND") or shutil.which("ph.x")
Q2R_CMD = shutil.which("q2r.x")
MATDYN_CMD = shutil.which("matdyn.x")

if not PW_CMD:  raise RuntimeError("pw.x not found — set QE_PW_COMMAND or add pw.x to PATH.")
if not PH_CMD:  raise RuntimeError("ph.x not found — set QE_PH_COMMAND or add ph.x to PATH.")
if not Q2R_CMD: raise RuntimeError("q2r.x not found in PATH.")
if not MATDYN_CMD: raise RuntimeError("matdyn.x not found in PATH.")

NPROCS = int(os.environ.get("NPROCS", "1")) #allocate more later!!

MPIRUN = shutil.which("mpirun") or shutil.which("mpiexec")

def run(cmd, cwd=None):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def mpicmd(exe, *args):
    if MPIRUN:
        return [MPIRUN, "-np", str(NPROCS), exe, *args]
    return [exe, *args]

# ---------- QE Inputs (tutorial-like) ----------
# ---------- QE Inputs (FAST PREVIEW) ----------
PREFIX = "GaAs"
CELLDM1 = 10.68

#cell params: A=3.94540, B=3.94540, C=16.09240

# Cutoffs (lighter)
ECUTWFC = 35           #  Ry, around 450 eV
ECUTRHO = 6 * ECUTWFC  # was 8*ecutwfc

# Coarser k-mesh for SCF
K_AUTOMATIC = (4, 4, 4, 0, 0, 0)  # 4x4x1 for uote

# Looser DFPT convergence + small q-mesh
TR2_PH = "1.0d-10"     # 1.0d-6
NQ = (2, 2, 2)         # 4x4x2

# set Γ-X–Γ-Y-Γ-Z
 
#BAND_PATH = [((0.000, 0.000, 0.000), 6, "G"),
#            ((1.000, 0.000, 0.000), 6, "X"),
#            ((0.000, 0.000, 0.000), 6, "G"),
#            ((0.000, 1.000, 0.000), 6, "Y"),
#            ((0.000, 0.000, 0.000), 6, "G"),
#            ((0.000, 0.000, 1.000), 6, "Z"),
#]


BAND_PATH = [
    ((0.0, 0.0, 0.0), 40, "G"),  # Γ
    ((0.5, 0.5, 0.0), 40, "X"),
    ((0.75,0.75,0.0), 40, "K"),
    ((0.0, 0.0, 0.0), 40, "G"),
    ((0.5, 0.5, 0.5), 40, "L"),
]


# Update filenames here to match the ones inside your SSSP folder: 
PSEUDOS = { "Ga": "ga_pbe_v1.4.uspp.F.UPF",
    "As": "as_pbe_v1.uspp.F.UPF", }
#PSEUDOS = {"U":,"Te":,"O":} #TBD

def ensure_setup():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    TMPDIR.mkdir(parents=True, exist_ok=True)
    for el, fname in PSEUDOS.items():
        f = PSEUDO_DIR / fname
        if not f.exists():
            raise FileNotFoundError(
                f"Missing pseudo for {el}: {f}\n"
                f"→ List your pseudos and update PSEUDOS[...] names above."
            )

def write_pw_scf():
    text = textwrap.dedent(f"""\
        &control
          calculation = 'scf'
          prefix = '{PREFIX}'
          pseudo_dir = '{PSEUDO_DIR.as_posix()}'
          outdir = '{TMPDIR.as_posix()}/'
          verbosity = 'high'
          wf_collect = .true.
        /
        &system
          ibrav = 2 
          celldm(1) = {CELLDM1}
          nat = 2 !12
          ntyp = 2 !3
          !nspin = 4
          !starting_magnetization(1)=3 ! (up 3, down-3 ,down-3, up 3)
          ecutwfc = {ECUTWFC}
          ecutrho = {ECUTRHO}
          !vdw_corr='grimme-d3'
          !occupations= 'smearing',
          !smearing='mp',
          !degauss=0.02,
        /
        &electrons
          mixing_beta = 0.7
          conv_thr = 1.0e-6      ! was 1.0e-8
          diagonalization = 'david'
        /

        /
        ATOMIC_SPECIES
          Ga 69.723    {PSEUDOS['Ga']}
          As 74.921595 {PSEUDOS['As']}
        ATOMIC_POSITIONS crystal
          Ga 0.00 0.00 0.00
          As 0.25 0.25 0.25
        K_POINTS automatic
          {K_AUTOMATIC[0]} {K_AUTOMATIC[1]} {K_AUTOMATIC[2]} {K_AUTOMATIC[3]} {K_AUTOMATIC[4]} {K_AUTOMATIC[5]}
    """)
    (WORKDIR / "pw.scf.in").write_text(text)

def write_ph_in():
    text = textwrap.dedent(f"""\
        &INPUTPH
          outdir = '{TMPDIR.as_posix()}/'
          prefix = '{PREFIX}'
          tr2_ph = {TR2_PH}
          ldisp = .true.
          nq1 = {NQ[0]}
          nq2 = {NQ[1]}
          nq3 = {NQ[2]}
          fildyn = '{PREFIX}.dyn'
          epsil = .false.   ! quick preview: no LO-TO splitting
        /
    """)
    (WORKDIR / "ph.in").write_text(text)

def write_q2r_in():
    text = textwrap.dedent(f"""\
        &INPUT
          fildyn = '{PREFIX}.dyn'
          zasr = 'crystal'
          flfrc = '{PREFIX}.fc'
        /
    """)
    (WORKDIR / "q2r.in").write_text(text)

def write_matdyn_in():
    lines = [str(len(BAND_PATH))]
    for (qx, qy, qz), nseg, label in BAND_PATH:
        lines.append(f"  {qx:.6f} {qy:.6f} {qz:.6f}  {nseg}  ! {label}")
    blk = "\n".join(lines)
    text = textwrap.dedent(f"""\
        &INPUT
          asr = 'crystal'
          flfrc = '{PREFIX}.fc'
          flfrq = '{PREFIX}.freq'
          flvec = '{PREFIX}.modes'
          q_in_band_form = .true.
          q_in_cryst_coord = .true.
        /
        {blk}
    """)
    (WORKDIR / "matdyn.in").write_text(text)

def scf():
    write_pw_scf()
    run(mpicmd(PW_CMD, "-i", "pw.scf.in"), cwd=WORKDIR)

def ph():
    write_ph_in()
    run(mpicmd(PH_CMD, "-i", "ph.in"), cwd=WORKDIR)

def q2r():
    write_q2r_in()
    # run serial; q2r.x is tiny
    run([Q2R_CMD, "-i", "q2r.in"], cwd=WORKDIR)

def matdyn():
    write_matdyn_in()
    # run serial; matdyn.x is light
    run([MATDYN_CMD, "-i", "matdyn.in"], cwd=WORKDIR)
def save_dispersion_data():
    """Save s-coordinate and phonon branches to .dat and .csv with metadata."""
    gp = WORKDIR / f"{PREFIX}.freq.gp"
    if not gp.exists():
        raise FileNotFoundError(f"Expected {gp} after matdyn.x")

    data = np.loadtxt(gp)
    s = data[:, 0]
    bands = data[:, 1:]   # columns: mode frequencies (cm^-1)

    # Build segment ticks with Matdyn's shared endpoints (no duplicated joints)
    seg_counts = [n for _, n, _ in BAND_PATH]
    tick_idx = [0]
    acc = 0
    for n in seg_counts[:-1]:  # stop before last segment
        acc += n
        tick_idx.append(acc - 1)
    tick_positions = [float(s[i]) for i in tick_idx]

    # Labels (use Γ for 'G')
    tick_labels = [lbl for *_, lbl in BAND_PATH]
    tick_labels = ['Γ' if L == 'G' else L for L in tick_labels]

    # ----- Write .dat (space-separated) with a header -----
    dat_path = WORKDIR / f"{PREFIX}_phonon_dispersion.dat"
    with open(dat_path, "w") as f:
        f.write("# Phonon dispersion (DFPT) exported from matdyn\n")
        f.write(f"# System: {PREFIX}\n")
        f.write(f"# Source file: {gp.name}\n")
        f.write("# Columns: s  w1(cm^-1)  w2  ...  w(3*nat)\n")
        f.write("# Path: " + " → ".join(tick_labels) + "\n")
        f.write("# Tick positions (s): " + " ".join(f"{x:.8f}" for x in tick_positions) + "\n")
        f.write("# Tick indices: " + " ".join(str(i) for i in tick_idx) + "\n")
        np.savetxt(f, np.column_stack([s, bands]), fmt="%.8f")

    # ----- Also write CSV (comma-separated) -----
    csv_path = WORKDIR / f"{PREFIX}_phonon_dispersion.csv"
    header = "s," + ",".join(f"w{i+1}_cm^-1" for i in range(bands.shape[1]))
    np.savetxt(csv_path, np.column_stack([s, bands]),
               delimiter=",", header=header, comments="", fmt="%.8f")

    print(f"Saved {dat_path} and {csv_path}")
    return s, bands, tick_idx, tick_positions, tick_labels


def plot_dispersion():
    # Save first, then plot using returned metadata
    s, bands, tick_idx, tick_positions, tick_labels = save_dispersion_data()

    plt.figure(figsize=(6, 4))
    for i in range(bands.shape[1]):
        plt.plot(s, bands[:, i], lw=1)

    # Ticks/lines
    label_for_matplotlib = [r'$\Gamma$' if L == 'Γ' else L for L in tick_labels]
    plt.xticks(tick_positions, label_for_matplotlib)
    for x in tick_positions:
        plt.axvline(x=x, lw=0.5)

    plt.ylabel("Frequency (cm$^{-1}$)")
    plt.title("GaAs phonon dispersion (DFPT)")
    plt.ylim(bottom=0)
    plt.tight_layout()
    outpng = WORKDIR / "GaAs_phonon_dispersion.png"
    plt.savefig(outpng, dpi=300)
    print(f"Saved {outpng}")


if __name__ == "__main__":
    ensure_setup()
    scf()
    ph()
    q2r()
    matdyn()
    plot_dispersion()
