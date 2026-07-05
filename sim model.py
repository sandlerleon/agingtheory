"""
SIM = Sugar Index Muscle
A minimal dynamical-systems model coupling epigenetic information loss (I),
muscle reserve (M) and adiposity (F), with glucose (G) at quasi-steady-state,
under a GLP-1 control input u. Produces the flip-point result, the "BMI lies"
demonstration, the two-tier (information-loss vs accessible-proxy) clock, and
the muscle-cost bifurcation of the optimal dose.

All parameters are ILLUSTRATIVE / semi-quantitative (dimensionless, life-course
time in years). The model demonstrates existence and coherence of the index and
its flip-point property WITHIN the framework; it is not fit to human data.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 140, "savefig.dpi": 160, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

# ---- colour palette (colourblind-friendly) ----
C = {"I": "#332288", "G": "#CC6677", "M": "#117733", "F": "#DDCC77",
     "SIM": "#AA4499", "BMI": "#88CCEE", "opt": "#882255"}

# ============================================================
#  PARAMETERS (illustrative)
# ============================================================
P = dict(
    G0=1.00,           # young reference glucose scale
    a_I=0.80, a_F=0.45, a_M=1.00, a_u=0.55,   # glucose QSS couplings
    r0=0.0020,         # baseline (glucose-independent) aging rate  [1/yr]
    r_g=0.0130,        # glucose-driven aging coefficient
    r_m=0.0400,        # muscle-DEFICIT aging channel (myokine/disposal/frailty)
    c_sig=0.90,        # susceptibility feedback (NAD/epigenetic reserve loss)
    lam=0.28,          # age-related muscle set-point decline (sarcopenia vs info-loss)
    kM=0.45,           # rate muscle relaxes toward its set-point [1/yr]
    phi=0.50,          # MAX fractional muscle loss under GLP-1 (bounded)
    ku=1.00,           # dose at half-maximal muscle loss (Hill n=2; KEY parameter)
    e=0.0,             # exercise / anabolic co-intervention rate
    M_max=1.00,
    gF=0.055,          # fat adjustment rate
    c_F=0.16,          # GLP-1 fat-loss rate
    nH=2.0,            # Hill exponent for dose-dependent muscle loss
    p_m=2.0,           # exponent of the muscle-deficit aging penalty
    kappa=1.00,        # SYSTEMIC COUPLING: how strongly local info-loss propagates
                       #   to systemic decline. Humans ~1; trees (compartmentalised) ->0
)

def glucose(I, M, F, u, p):
    return p["G0"] * (1 + p["kappa"]*p["a_I"]*I + p["a_F"]*F) / (1 + p["a_M"]*M + p["a_u"]*u)

def sigma(I, p):
    return 1 + p["kappa"]*p["c_sig"]*I

def F_setpoint(I, p):
    return 0.40 + p["kappa"]*0.40*I   # adiposity drift with age is a systemic (coupled) effect

def M_target(u, I, p):
    # muscle set-point = (age-related decline) x (bounded sigmoidal GLP-1 loss, Hill n=2),
    # lifted by any anabolic/exercise co-intervention e.
    base = (1.0 - p["lam"]*I) * (1.0 - p["phi"]*u**p["nH"]/(p["ku"]**p["nH"] + u**p["nH"]))
    return base + p["e"]*(p["M_max"] - base)

def rhs(t, y, u_func, p):
    I, M, F = y
    u = u_func(t)
    G = glucose(I, M, F, u, p)
    # baseline (turnover) info-loss ALWAYS accrues -> the "clock". Systemic amplifiers
    # (glycation, frailty feedback) are gated by the coupling coefficient kappa.
    defc = max(1.0 - M, 0.0)   # muscle deficit (guard against tiny numerical M>1)
    dI = (p["r0"] + p["kappa"]*(p["r_g"]*G*sigma(I, p) + p["r_m"]*defc**p["p_m"])) * (1 - I)
    dM = p["kM"]*(M_target(u, I, p) - M)
    dF = p["gF"]*(F_setpoint(I, p) - F) - p["c_F"]*u*F
    return [dI, dM, dF]

def simulate(u_func, p, t_end=50.0, y0=(0.10, 1.00, 0.40), n=1001):
    t = np.linspace(0, t_end, n)
    sol = solve_ivp(rhs, (0, t_end), list(y0), t_eval=t, args=(u_func, p),
                    method="LSODA", rtol=1e-8, atol=1e-10)
    I, M, F = sol.y
    G = glucose(I, M, F, np.array([u_func(tt) for tt in t]), p)
    SIM = G * sigma(I, p) / M
    SIM_simple = G / M
    BMI = 24.0 * (F + M) / (0.40 + 1.00)          # normalised to young ~24
    return dict(t=t, age=25+t, I=I, M=M, F=F, G=G, SIM=SIM,
                SIM_simple=SIM_simple, BMI=BMI)

const = lambda c: (lambda t: c)

# reference (untreated) SIM trajectory  ->  metabolic-age calibration
base = simulate(const(0.0), P)
def metabolic_age(sim_value):
    """invert the untreated SIM(age) curve to read a metabolic age (yrs)."""
    return np.interp(sim_value, base["SIM"], base["age"])

# ============================================================
#  FIGURE 1 — why sugar rises with age (untreated mechanism)
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
ax[0].plot(base["age"], base["I"], color=C["I"], lw=2.2, label="Information loss $I$")
ax[0].plot(base["age"], base["M"], color=C["M"], lw=2.2, label="Muscle reserve $M$")
ax[0].plot(base["age"], base["F"], color=C["F"], lw=2.2, label="Adiposity $F$")
ax[0].set_xlabel("Chronological age (yr)"); ax[0].set_ylabel("State (normalised)")
ax[0].set_title("A  State variables over the life course"); ax[0].legend(loc="center left")
ax[1].plot(base["age"], base["G"], color=C["G"], lw=2.4)
ax[1].set_xlabel("Chronological age (yr)"); ax[1].set_ylabel("Glycaemic burden $G$ (norm.)")
ax[1].set_title("B  Glucose rises as information is lost")
ax[1].annotate("driven by rising $I$\nand falling $M$", xy=(60, base["G"][np.argmin(abs(base['age']-60))]),
               xytext=(38, base["G"].max()*0.985), fontsize=9,
               arrowprops=dict(arrowstyle="->", color="0.4"))
pct = 100*(base["G"][-1]/base["G"][0]-1)
fig.suptitle(f"Figure 1.  Mechanism: glycaemic burden rises {pct:.0f}% across the life course as an emergent output of information loss",
             fontsize=10.5, y=1.02)
plt.tight_layout(); plt.savefig("/home/claude/fig1_mechanism.png", bbox_inches="tight"); plt.close()

# ============================================================
#  FIGURE 2 — the flip-point (dose sweep)
#  Treat with constant u from age 45 (t=20) to 75 (t=50).
# ============================================================
t_start = 20.0
def dose_from(c, t0=t_start):
    return lambda t: (c if t >= t0 else 0.0)

us = np.linspace(0, 2.0, 61)
I_final, M_final, MA_final = [], [], []
for u in us:
    r = simulate(dose_from(u), P)
    I_final.append(r["I"][-1]); M_final.append(r["M"][-1])
    MA_final.append(metabolic_age(r["SIM"][-1]))
I_final, M_final, MA_final = map(np.array, (I_final, M_final, MA_final))
u_star = us[np.argmin(I_final)]
u_star_MA = us[np.argmin(MA_final)]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
ax[0].plot(us, I_final, color=C["I"], lw=2.4)
ax[0].axvline(u_star, color=C["opt"], ls="--", lw=1.6)
ax[0].scatter([u_star],[I_final.min()], color=C["opt"], zorder=5)
ax[0].annotate(f"flip-point\n$u^*$≈{u_star:.2f}", xy=(u_star, I_final.min()),
               xytext=(u_star+0.35, I_final.min()+ (I_final.max()-I_final.min())*0.35),
               fontsize=9, arrowprops=dict(arrowstyle="->", color=C["opt"]))
ax[0].fill_betweenx([I_final.min(), I_final.max()], 0, u_star, color=C["M"], alpha=0.06)
ax[0].fill_betweenx([I_final.min(), I_final.max()], u_star, 2.0, color=C["G"], alpha=0.06)
ax[0].text(u_star*0.5, I_final.max(), "under-\ntreated", ha="center", va="top", fontsize=8.5, color=C["M"])
ax[0].text((u_star+2)/2, I_final.max(), "muscle-cost\nbackfire", ha="center", va="top", fontsize=8.5, color=C["G"])
ax[0].set_xlabel("GLP-1 dose $u$ (constant, from age 45)")
ax[0].set_ylabel("Information loss at age 75  $I(T)$")
ax[0].set_title("A  Aging outcome is U-shaped in dose")

ax2 = ax[1]; ax2b = ax2.twinx()
ax2.plot(us, MA_final, color=C["SIM"], lw=2.4, label="Metabolic age at 75")
ax2b.plot(us, M_final, color=C["M"], lw=2.0, ls=":", label="Muscle at 75")
ax2.axvline(u_star_MA, color=C["opt"], ls="--", lw=1.4)
ax2.set_xlabel("GLP-1 dose $u$"); ax2.set_ylabel("Metabolic age (yr)", color=C["SIM"])
ax2b.set_ylabel("Muscle reserve $M(T)$", color=C["M"])
ax2.set_title("B  Optimal dose trades glucose gain vs muscle cost")
ax2.tick_params(axis="y", colors=C["SIM"]); ax2b.tick_params(axis="y", colors=C["M"])
fig.suptitle("Figure 2.  The muscle-cost flip-point: below $u^*$ the intervention under-treats, above $u^*$ muscle loss backfires",
             fontsize=10.5, y=1.02)
plt.tight_layout(); plt.savefig("/home/claude/fig2_flippoint.png", bbox_inches="tight"); plt.close()

# ============================================================
#  FIGURE 3 — "BMI lies, SIM tells the truth"
#  Fixed supratherapeutic course from age 45.
# ============================================================
u_supra = 1.50
r = simulate(dose_from(u_supra), P)
mask = r["age"] >= 45
age_t = r["age"][mask]
ref_SIM = np.interp(45, base["age"], base["SIM"])   # untreated value at treatment start
ref_BMI = np.interp(45, base["age"], base["BMI"])
BMIn = r["BMI"][mask] / ref_BMI
SIMn = r["SIM"][mask] / ref_SIM
turn = age_t[np.argmin(SIMn)]

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(age_t, BMIn, color=C["BMI"], lw=2.6, label="BMI proxy (normalised)")
ax.plot(age_t, SIMn, color=C["SIM"], lw=2.6, label="SIM index (normalised)")
ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
ax.axvline(turn, color=C["opt"], ls="--", lw=1.4)
ax.scatter([turn], [SIMn.min()], color=C["opt"], zorder=5)
ax.annotate("SIM turns:\nmuscle loss overtakes\nglucose gain (flip-point)",
            xy=(turn, SIMn.min()), xytext=(turn+1.2, 0.93),
            fontsize=9, arrowprops=dict(arrowstyle="->", color=C["opt"]))
ax.annotate("BMI keeps falling\n→ reads 'drug working'",
            xy=(age_t[-1], BMIn[-1]), xytext=(58, BMIn[-1]-0.14), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="0.4"))
ax.set_xlabel("Age during treatment (yr)"); ax.set_ylabel("Index (relative to treatment start)")
ax.set_title("Figure 3.  BMI declines monotonically while SIM reveals the flip-point")
ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig("/home/claude/fig3_bmi_lies.png", bbox_inches="tight"); plt.close()

# ============================================================
#  FIGURE 4 — two-tier clock + exercise rescue
# ============================================================
# entropy-clock analog = I(t) (direct info-loss);  accessible proxy = SIM(t)
P_ex = dict(P); P_ex["e"] = 0.030            # add anabolic/exercise co-intervention
r_notx = simulate(const(0.0), P)
r_glp  = simulate(dose_from(u_supra), P)
r_glpex= simulate(dose_from(u_supra), P_ex)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
# left: SIM tracks I across regimes
def z(x): return (x - x.min())/(x.max()-x.min())
ax[0].plot(r_glp["age"], z(r_glp["I"]),  color=C["I"], lw=2.2, label="Information loss $I$ (entropy-clock analog)")
ax[0].plot(r_glp["age"], z(r_glp["SIM"]),color=C["SIM"], lw=2.2, ls="--", label="SIM (accessible proxy)")
ax[0].set_xlabel("Age (yr)"); ax[0].set_ylabel("Normalised reading")
ax[0].set_title("A  Accessible proxy tracks the direct clock")
ax[0].legend(loc="upper left", fontsize=8.5)
# right: exercise rescue of the flip-point
m2 = r_glp["age"] >= 45
ax[1].plot(r_glp["age"][m2],  (r_glp["SIM"]/r_glp["SIM"][m2][0])[m2],   color=C["SIM"], lw=2.4, label="GLP-1 alone")
ax[1].plot(r_glpex["age"][m2],(r_glpex["SIM"]/r_glpex["SIM"][m2][0])[m2],color=C["M"], lw=2.4, label="GLP-1 + muscle preservation")
ax[1].axhline(1.0, color="0.7", lw=0.8, ls=":")
ax[1].set_xlabel("Age during treatment (yr)"); ax[1].set_ylabel("SIM (rel. to start)")
ax[1].set_title("B  Preserving muscle abolishes the flip-point")
ax[1].legend(loc="lower left", fontsize=9)
fig.suptitle("Figure 4.  Two-tier clock: a cheap metabolic proxy shadows the direct information-loss readout, and combination therapy holds SIM down",
             fontsize=10.0, y=1.03)
plt.tight_layout(); plt.savefig("/home/claude/fig4_twotier.png", bbox_inches="tight"); plt.close()

# ============================================================
#  FIGURE 5 — flip-point (optimal dose) bifurcates with muscle-cost
# ============================================================
phis = np.linspace(0.15, 0.65, 26)
u_opt = []
for ph in phis:
    Pv = dict(P); Pv["phi"] = ph
    vals = [simulate(dose_from(u), Pv)["I"][-1] for u in us]
    u_opt.append(us[int(np.argmin(vals))])
u_opt = np.array(u_opt)

fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.plot(phis, u_opt, color=C["opt"], lw=2.8)
ax.fill_between(phis, u_opt, 2.0, color=C["G"], alpha=0.07)
ax.fill_between(phis, 0, u_opt, color=C["M"], alpha=0.07)
ax.set_xlabel("Muscle-loss magnitude $\\phi$  (max fractional lean mass lost)")
ax.set_ylabel("Flip-point / optimal dose $u^*$")
ax.set_title("Figure 5.  Muscle-sparing regimens widen the safe dose window")
ax.text(0.18, 1.55, "muscle-sparing\n(low $\\phi$): dose higher\nbefore backfire", color=C["M"], fontsize=8.8)
ax.text(0.44, 0.28, "muscle-costly (high $\\phi$):\nflip-point at low dose", color=C["G"], fontsize=8.8)
ax.set_ylim(0, 2.0)
ax.annotate("combination therapy\nmoves you left", xy=(0.30, np.interp(0.30, phis, u_opt)),
            xytext=(0.34, 1.15), fontsize=9, color=C["opt"],
            arrowprops=dict(arrowstyle="->", color=C["opt"]))
plt.tight_layout(); plt.savefig("/home/claude/fig5_bifurcation.png", bbox_inches="tight"); plt.close()

# ============================================================
#  FIGURE 6 — systemic coupling kappa: the "tree" limit
#  Local information-loss clock ticks regardless; systemic aging needs coupling.
# ============================================================
regimes = [(0.03, "Tree-like ($\\kappa\\approx0$)", C["M"]),
           (0.35, "Intermediate", C["F"]),
           (1.00, "Human ($\\kappa=1$)", C["G"])]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
for k, lab, col in regimes:
    Pk = dict(P); Pk["kappa"] = k
    r = simulate(const(0.0), Pk)
    ax[0].plot(r["age"], r["I"], color=col, lw=2.4, label=lab)
    ax[1].plot(r["age"], metabolic_age(r["SIM"]), color=col, lw=2.4, label=lab)
ax[0].plot([25, 75], [25, 75], "k:", lw=0.8)  # (visual ref not used)
ax[0].lines[-1].remove()
ax[0].set_xlabel("Chronological age (yr)"); ax[0].set_ylabel("Local information loss $I$")
ax[0].set_title("A  The information-loss clock ticks\nin every regime"); ax[0].legend(fontsize=8.5, loc="upper left")
ax[1].plot([25, 75], [25, 75], "k:", lw=1.0, label="chronological")
ax[1].set_xlabel("Chronological age (yr)"); ax[1].set_ylabel("Systemic metabolic age (yr)")
ax[1].set_title("B  Systemic aging appears only\nwhen coupling is high"); ax[1].legend(fontsize=8.5, loc="upper left")

# kappa sweep at age 75
kappas = np.linspace(0.0, 1.2, 49)
I_T, MA_T = [], []
for k in kappas:
    Pk = dict(P); Pk["kappa"] = k
    r = simulate(const(0.0), Pk)
    I_T.append(r["I"][-1]); MA_T.append(metabolic_age(r["SIM"][-1]))
I_T, MA_T = np.array(I_T), np.array(MA_T)
ax2 = ax[2]; ax2b = ax2.twinx()
l1, = ax2.plot(kappas, MA_T, color=C["SIM"], lw=2.8, label="Systemic metabolic age")
l2, = ax2b.plot(kappas, I_T, color=C["I"], lw=2.4, ls="--", label="Local info-loss clock $I(T)$")
ax2.axvspan(0.0, 0.15, color=C["M"], alpha=0.10)
ax2.text(0.02, MA_T.max()*0.96, "tree\nregime", color=C["M"], fontsize=8.5, va="top")
ax2.set_xlabel("Systemic coupling $\\kappa$"); ax2.set_ylabel("Systemic metabolic age at 75 (yr)", color=C["SIM"])
ax2b.set_ylabel("Local information loss $I(T)$", color=C["I"])
ax2.tick_params(axis="y", colors=C["SIM"]); ax2b.tick_params(axis="y", colors=C["I"])
ax2.set_title("C  Decoupling: systemic age collapses\nas $\\kappa\\to0$; the clock does not")
ax2.legend(handles=[l1, l2], fontsize=8.5, loc="center right")
fig.suptitle("Figure 6.  Systemic aging = information loss $\\times$ coupling. Trees ($\\kappa\\to0$) accrue the information-loss clock without systemic senescence.",
             fontsize=10.5, y=1.04)
plt.tight_layout(); plt.savefig("/home/claude/fig6_coupling.png", bbox_inches="tight"); plt.close()

# ============================================================
#  NUMERIC SUMMARY  (printed -> used in the paper)
# ============================================================
print("=== SIM model — key results ===")
print(f"Glucose rise across life course (untreated): {pct:.1f}%")
print(f"Info loss I: {base['I'][0]:.2f} -> {base['I'][-1]:.2f}")
print(f"Muscle M:    {base['M'][0]:.2f} -> {base['M'][-1]:.2f}")
print(f"Flip-point dose u* (min info loss): {u_star:.3f}")
print(f"Flip-point dose u* (min metabolic age): {u_star_MA:.3f}")
print(f"Metabolic age at 75, untreated:  {metabolic_age(base['SIM'][-1]):.1f} yr")
print(f"Metabolic age at 75, u=u*:       {MA_final[np.argmin(abs(us-u_star))]:.1f} yr")
print(f"Metabolic age at 75, u=1.55:     {metabolic_age(simulate(dose_from(1.55),P)['SIM'][-1]):.1f} yr")
print(f"SIM turning age (u=1.55 course):  {turn:.1f} yr")
print(f"Flip-point dose range over phi sweep: {u_opt.min():.2f}..{u_opt.max():.2f}  (phi {phis.min():.2f}..{phis.max():.2f})")
print("--- kappa (coupling) decoupling ---")
print(f"kappa=0.03 (tree): I(75)={I_T[np.argmin(abs(kappas-0.03))]:.2f}, metabolic age={MA_T[np.argmin(abs(kappas-0.03))]:.1f} yr")
print(f"kappa=1.00 (human): I(75)={I_T[-9]:.2f}, metabolic age={MA_T[np.argmin(abs(kappas-1.0))]:.1f} yr")
print(f"Local clock rises {I_T[-1]/I_T[0]:.1f}x across kappa; systemic age spans {MA_T.min():.0f}-{MA_T.max():.0f} yr")
print("Figures written: fig1..fig6")
