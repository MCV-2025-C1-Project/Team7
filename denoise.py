import csv
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# === CONFIG ===
Q_DIR = Path("./datasets/qsd1_w3")
OUTIMG = Path("./denoise_out")
OUTIMG.mkdir(exist_ok=True, parents=True)
PLOTS = Path("./denoise_plots")
PLOTS.mkdir(exist_ok=True, parents=True)
SFX = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TOP_N_FORCE = 4  # força plots dels TOP-N sospitosos per mètrica
DPI = 220


# ---------- Mètriques (robustes) ----------
def lap_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var()


def sp_ratio(gray):
    # % exactes 0/255 → soroll impuls "dur"
    return float(((gray == 0) | (gray == 255)).mean())


def impulse_residue(gray):
    # Diferència després de median 3x3 (caça impulsos no exactes)
    med = cv2.medianBlur(gray, 3)
    return float(np.mean(np.abs(gray.astype(np.int16) - med.astype(np.int16))) / 255.0)


def sigma_on_weak_texture(gray, grad_th=6.0):
    # σ estimat només a zones planes per no confondre textura amb soroll
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy)
    mask = g < grad_th
    if np.count_nonzero(mask) < 0.05 * gray.size:
        mask = g < (grad_th * 1.5)
    vals = gray[mask] if np.count_nonzero(mask) else gray.reshape(-1)
    return float(np.std(vals))


def hp_energy(gray):
    # Energia high-pass (LoG suau)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    hp = cv2.filter2D(gray.astype(np.float32), -1, k)
    return float(np.mean(np.abs(hp)))


# ---------- Denoise helpers ----------
def nlm_colored(bgr, h=8, hColor=8, tw=7, sw=21):
    return cv2.fastNlMeansDenoisingColored(
        bgr, None, h=h, hColor=hColor, templateWindowSize=tw, searchWindowSize=sw
    )


def bilateral(bgr, d=7, sC=25, sS=7):
    return cv2.bilateralFilter(bgr, d=d, sigmaColor=sC, sigmaSpace=sS)


def guided_like(bgr):
    # “Pobre” de guided filter: bilateral + lleu sharpen per recuperar vores
    den = bilateral(bgr, d=5, sC=20, sS=5)
    sharp = cv2.addWeighted(den, 1.0, cv2.GaussianBlur(den, (0, 0), 1.0), -0.15, 0)
    return sharp


# ---------- 1) Escora totes les imatges i calcula mètriques ----------
rows, imgs = [], []
for p in sorted(Q_DIR.glob("*")):
    if p.suffix.lower() not in SFX:
        continue
    img = cv2.imread(str(p))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = {
        "name": p.name,
        "sp": sp_ratio(gray),
        "res": impulse_residue(gray),
        "sig": sigma_on_weak_texture(gray),
        "hp": hp_energy(gray),
        "sharp": lap_var(gray),
        "path": str(p),
    }
    rows.append(r)
    imgs.append((p, img, gray))

#if not rows:
#    print("[WARN] No s'han trobat imatges.")
#else:
#    print(f"[INFO] Imatges trobades: {len(rows)}")


# ---------- 2) Llindars adaptatius per percentils (+mínims absoluts) ----------
def pctl(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else 0.0


sp_arr = np.array([r["sp"] for r in rows])
res_arr = np.array([r["res"] for r in rows])
sig_arr = np.array([r["sig"] for r in rows])
hp_arr = np.array([r["hp"] for r in rows])

p90_sp, p90_res, p90_sig, p90_hp = (
    pctl(sp_arr, 90),
    pctl(res_arr, 90),
    pctl(sig_arr, 90),
    pctl(hp_arr, 90),
)

# Mínims absoluts “safety” (per si el lot és molt net)
ABS_SP_MIN, ABS_RES_MIN, ABS_SIG_MIN, ABS_HP_MIN = 0.002, 0.010, 3.0, 1.5
TH_SP = max(p90_sp, ABS_SP_MIN)
TH_RES = max(p90_res, ABS_RES_MIN)
TH_SIG = max(p90_sig, ABS_SIG_MIN)
TH_HP = max(p90_hp, ABS_HP_MIN)

#print(
#    f"[INFO] Percentils p90 -> sp:{p90_sp:.4f}, res:{p90_res:.4f}, sig:{p90_sig:.2f}, hp:{p90_hp:.2f}"
#)
#print(
#    f"[INFO] Llindars finals -> SP:{TH_SP:.4f} RES:{TH_RES:.4f} SIG:{TH_SIG:.2f} HP:{TH_HP:.2f}"
#)

# Desa CSV de mètriques
with open(PLOTS / "noise_scores.csv", "w", newline="") as f:
    w = csv.DictWriter(
        f, fieldnames=["name", "sp", "res", "sig", "hp", "sharp", "path"]
    )
    w.writeheader()
    w.writerows(rows)
#print(f"[INFO] CSV -> {PLOTS / 'noise_scores.csv'}")


# ---------- 3) Funció de decisió + denoise escalonat ----------
def decide_and_denoise(bgr, metrics):
    """
    Regles:
    - SP o RES alts → median 3x3; si RES segueix alt → 5x5; després millora vores amb bilateral suau.
    - SIG o HP alts → NLM; si la imatge és molt nítida → prova bilateral/guided_like i tria el que preserva més Laplacià.
    - Salvaguarda: no permetre caiguda de nitidesa > 20% (Laplacià).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp0 = lap_var(gray)
    reasons = []
    if metrics["sp"] >= TH_SP:
        reasons.append("SP")
    if metrics["res"] >= TH_RES:
        reasons.append("RES")
    if metrics["sig"] >= TH_SIG:
        reasons.append("SIG")
    if metrics["hp"] >= TH_HP:
        reasons.append("HP")

    if not reasons:
        return bgr, "skip", sharp0, sharp0

    out, choice = bgr.copy(), None

    # Cas impulsos (salt-&-pepper o residu alt)
    if ("SP" in reasons) or ("RES" in reasons):
        out = cv2.medianBlur(out, 3)
        choice = "median 3x3"
        res_after = impulse_residue(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))
        if res_after > TH_RES:
            out = cv2.medianBlur(out, 5)
            choice = "median 5x5"
        # recuperar vores una mica
        out = bilateral(out, d=5, sC=20, sS=5)
        choice += " + bilateral"

    # Cas gaussian/alta-freq
    elif ("SIG" in reasons) or ("HP" in reasons):
        cand = []
        # NLM suau
        nlm = nlm_colored(bgr, h=8, hColor=8, tw=7, sw=21)
        cand.append(("NLM h8", nlm))
        # si molt nítida, prova alternatives i tria la millor per Laplacià post
        if sharp0 > 180.0:
            bil = bilateral(bgr, d=7, sC=25, sS=7)
            cand.append(("bilateral d7 sC25 sS7", bil))
            gui = guided_like(bgr)
            cand.append(("guided-like", gui))
        # Escull pel millor compromís: Laplacià alt i residu baix respecte original
        best, best_name, best_score = None, None, -1e9
        for name, im in cand:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            sh = lap_var(g)
            # penalitza oversmooth i massa diferència
            diff = (
                float(np.mean(np.abs(g.astype(np.int16) - gray.astype(np.int16))))
                / 255.0
            )
            score = sh - 50.0 * diff
            if score > best_score:
                best, best_name, best_score = im, name, score
        out, choice = best, best_name

    sharp1 = lap_var(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))
    if sharp1 < 0.8 * sharp0:  # salvaguarda
        return bgr, "reverted (oversmooth)", sharp0, sharp0
    return out, choice, sharp0, sharp1


# ---------- 4) Força TOP-N per assegurar exemples als plots ----------
def top_ids(arr, n):
    return np.argsort(-arr)[:n].tolist()


force_idx = set(
    top_ids(sp_arr, TOP_N_FORCE)
    + top_ids(res_arr, TOP_N_FORCE)
    + top_ids(sig_arr, TOP_N_FORCE)
    + top_ids(hp_arr, TOP_N_FORCE)
)

# ---------- 5) Procés principal + plots ----------
for i, (p, img, gray) in enumerate(imgs):
    metrics = {
        "sp": rows[i]["sp"],
        "res": rows[i]["res"],
        "sig": rows[i]["sig"],
        "hp": rows[i]["hp"],
    }
    out, choice, sharp0, sharp1 = decide_and_denoise(img, metrics)
    did_filter = (choice != "skip") and ("reverted" not in choice)

    # Desa la imatge denoised si s'ha aplicat filtrat
    if did_filter:
        cv2.imwrite(str(OUTIMG / f"{p.stem}__{choice.replace(' ', '_')}.jpg"), out)

    # Plota si s'ha filtrat O si és un dels forçats
    if did_filter or (i in force_idx):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0].axis("off")
        ax[0].set_title(
            f"{p.name}\nsp={metrics['sp']:.4f} res={metrics['res']:.4f}\nσweak={metrics['sig']:.2f} hp={metrics['hp']:.2f}"
        )
        ax[1].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        ax[1].axis("off")
        ax[1].set_title(f"Denoised: {choice}\nsharp {sharp0:.1f}→{sharp1:.1f}")
        plt.tight_layout()
        plt.savefig(
            PLOTS
            / f"{p.stem}__{('FORCED_' if not did_filter else '')}{choice.replace(' ', '_')}.png",
            dpi=DPI,
        )
        plt.close(fig)

def denoise_image(img: np.ndarray) -> np.ndarray:
    # STEP 1: Escora totes les imatges i calcula mètriques
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = {
        "sp": sp_ratio(gray),
        "res": impulse_residue(gray),
        "sig": sigma_on_weak_texture(gray),
        "hp": hp_energy(gray),
        "sharp": lap_var(gray)
    }
    #rows.append(r)
    #imgs.append((p, img, gray))
    
    # STEP 2: Llindars adaptatius per percentils (+mínims absoluts)
    """
    sp_arr = np.array(r["sp"])
    res_arr = np.array(r["res"])
    sig_arr = np.array(r["sig"])
    hp_arr = np.array(r["hp"])

    p90_sp, p90_res, p90_sig, p90_hp = (
        pctl(sp_arr, 90),
        pctl(res_arr, 90),
        pctl(sig_arr, 90),
        pctl(hp_arr, 90),
    )


    ABS_SP_MIN, ABS_RES_MIN, ABS_SIG_MIN, ABS_HP_MIN = 0.002, 0.010, 3.0, 1.5
    TH_SP = max(p90_sp, ABS_SP_MIN)
    TH_RES = max(p90_res, ABS_RES_MIN)
    TH_SIG = max(p90_sig, ABS_SIG_MIN)
    TH_HP = max(p90_hp, ABS_HP_MIN)

    # STEP 4: Força TOP-N per assegurar exemples als plots
    force_idx = set(
        top_ids(sp_arr, TOP_N_FORCE)
        + top_ids(res_arr, TOP_N_FORCE)
        + top_ids(sig_arr, TOP_N_FORCE)
        + top_ids(hp_arr, TOP_N_FORCE)
    )
    """
    # ---------- 5) Procés principal + plots ----------
    metrics = {
        "sp": r["sp"],
        "res": r["res"],
        "sig": r["sig"],
        "hp": r["hp"],
    }
    out, choice, sharp0, sharp1 = decide_and_denoise(img, metrics)
    did_filter = (choice != "skip") and ("reverted" not in choice)
    
    """
    # Desa la imatge denoised si s'ha aplicat filtrat
    if did_filter:
        cv2.imwrite(str(OUTIMG / f"{p.stem}__{choice.replace(' ', '_')}.jpg"), out)

    # Plota si s'ha filtrat O si és un dels forçats
    if did_filter or (i in force_idx):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0].axis("off")
        ax[0].set_title(
            f"{p.name}\nsp={metrics['sp']:.4f} res={metrics['res']:.4f}\nσweak={metrics['sig']:.2f} hp={metrics['hp']:.2f}"
        )
        ax[1].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        ax[1].axis("off")
        ax[1].set_title(f"Denoised: {choice}\nsharp {sharp0:.1f}→{sharp1:.1f}")
        plt.tight_layout()
        plt.savefig(
            PLOTS
            / f"{p.stem}__{('FORCED_' if not did_filter else '')}{choice.replace(' ', '_')}.png",
            dpi=DPI,
        )
        plt.close(fig)
        print(f"[PLOT] {p.name} -> {choice} (forced={i in force_idx})")
    else:
        print(f"[SKIP] {p.name} -> skip")

    print("✅ Fet! Plots a:", PLOTS, " | Denoised a:", OUTIMG)
    
    """
    return out
