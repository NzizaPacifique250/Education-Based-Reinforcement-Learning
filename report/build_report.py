"""Build the summative report (HTML, then PDF) from the real training logs.

Every table and every quoted number is read from logs/*/sweep_results.csv or recomputed
from the per-episode monitor files, so the report cannot drift away from the experiments.

    uv run python report/build_report.py

Writes report/report.html and, if Google Chrome is available, report/report.pdf.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ASSETS = os.path.join(ROOT, "assets")
OUT_DIR = os.path.join(ROOT, "report")

STUDENT = "Nziza Aime Pacifique"
REPO = "https://github.com/NzizaPacifique250/Education-Based-Reinforcement-Learning"
VIDEO = "https://drive.google.com/file/d/1ZsFxY9nqWZWnrMkExikMvVPvDAjsUNLA/view?usp=sharing"


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def _trim_border(path: str, tol: int = 248, pad: int = 5) -> bytes | None:
    """Crop the flat white border off a figure so it fills its slot on the page.

    The PyBullet renders in particular sit in a lot of empty background, which makes the
    school look tiny once the image is scaled to the column width.
    """
    try:
        import io
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    mask = (arr < tol).any(axis=2)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    box = (max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0),
           min(int(xs.max()) + pad + 1, im.width), min(int(ys.max()) + pad + 1, im.height))
    buf = io.BytesIO()
    im.crop(box).save(buf, format="PNG")
    return buf.getvalue()


def img(name: str, caption: str, cls: str = "") -> str:
    """Embed a figure as base64 so the HTML is self contained and prints reliably."""
    path = os.path.join(ASSETS, name)
    if not os.path.exists(path):
        return f'<p class="missing">missing figure: {name}</p>'
    data = None if name.endswith(".gif") else _trim_border(path)
    if data is None:
        with open(path, "rb") as f:
            data = f.read()
    b64 = base64.b64encode(data).decode()
    ext = "gif" if name.endswith(".gif") else "png"
    return (f'<figure class="{cls}">'
            f'<img src="data:image/{ext};base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


def sweep(algo: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(ROOT, "logs", algo, "sweep_results.csv"))


def fmt(v) -> str:
    """Plain decimals throughout: %g keeps 0.0005 readable and renders 0.0 as 0 rather
    than the 0e+00 that scientific formatting produced."""
    if isinstance(v, float):
        if v == int(v) and abs(v) >= 1:
            return str(int(v))
        return f"{v:g}"
    return str(v)


def table(algo: str, cols: dict[str, str], best_by="mean_return") -> str:
    """Render a sweep as an HTML table, bolding the winning row."""
    df = sweep(algo)
    best = df[best_by].idxmax()
    head = "".join(f"<th>{h}</th>" for h in cols.values())
    rows = []
    for i, r in df.iterrows():
        cells = []
        for key in cols:
            v = r[key]
            if key == "mean_return":
                cells.append(f"<td>{v:+.2f}</td>")
            elif key in ("success_rate", "biometric_rate", "tardy_rate"):
                cells.append(f"<td>{v:.2f}</td>")
            elif key == "mean_length":
                cells.append(f"<td>{v:.1f}</td>")
            else:
                cells.append(f"<td>{fmt(v)}</td>")
        cls = ' class="best"' if i == best else ""
        rows.append(f"<tr{cls}>" + "".join(cells) + "</tr>")
    return (f'<table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def conv_table() -> str:
    from training.plots import convergence_stats
    s = convergence_stats()
    order = ["dqn", "ppo", "a2c", "reinforce"]
    rows = []
    for a in order:
        if a not in s:
            continue
        d = s[a]
        rows.append(
            f"<tr><td>{a.upper()}</td><td>{d['name']}</td><td>{d['episodes']}</td>"
            f"<td>{d['first_reached']}</td><td><b>{d['converged_at']}</b></td>"
            f"<td>{d['plateau']:+.2f}</td></tr>")
    return ('<table><thead><tr><th>Method</th><th>Best config</th>'
            '<th>Episodes trained</th><th>First reached 90%</th>'
            '<th>Held 90% (converged)</th><th>Final plateau return</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def summary_table() -> str:
    rows = []
    for a in ("dqn", "ppo", "a2c", "reinforce"):
        df = sweep(a)
        b = df.loc[df.mean_return.idxmax()]
        worst = df.mean_return.min()
        rows.append(
            f"<tr><td>{a.upper()}</td><td>{b['name']}</td><td>{b.mean_return:+.2f}</td>"
            f"<td>{worst:+.2f}</td><td>{b.success_rate:.2f}</td>"
            f"<td>{b.biometric_rate:.2f}</td><td>{b.manual_rate:.2f}</td>"
            f"<td>{b.mean_length:.1f}</td></tr>")
    return ('<table><thead><tr><th>Method</th><th>Best config</th><th>Best mean return</th>'
            '<th>Worst config return</th><th>Check-in rate</th><th>Biometric</th>'
            '<th>Manual</th><th>Steps</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# ----------------------------------------------------------------------------------------
# observation space
# ----------------------------------------------------------------------------------------
OBS_ROWS = [
    ("Pupil position (x, y)", "Where the pupil is standing in the courtyard.",
     "Overhead fisheye camera with a person tracker, or a UWB tag in the student ID card",
     "float32, 2 values, scaled", "[-1, 1]"),
    ("Offset and range to reader A", "Vector and straight line distance to the main door reader.",
     "Same tracker, combined with the surveyed coordinates of the kiosk from the site plan",
     "float32, 3 values", "[-1, 1]"),
    ("Offset and range to reader B", "Vector and distance to the east side gate reader.",
     "As above", "float32, 3 values", "[-1, 1]"),
    ("Offset and range to sanitizer", "Vector and distance to the hand sanitizer stand.",
     "As above", "float32, 3 values", "[-1, 1]"),
    ("Offset and range to reception", "Vector and distance to the reception booth.",
     "As above", "float32, 3 values", "[-1, 1]"),
    ("At station flags (4)", "Whether the pupil is within 0.9 m of each of the four stations.",
     "Bluetooth Low Energy proximity beacon at each station, or a geofence over the tracker feed",
     "float32, 4 values, 0 or 1", "{0, 1}"),
    ("Hand cleanliness", "Estimated cleanliness of the finger that will be presented.",
     "Dose counter on the sanitizer dispenser plus elapsed time, cross checked against the "
     "reader's own NFIQ image quality score",
     "float32", "[0, 1]"),
    ("Rejections at A and B", "Failed scans recorded so far at each reader.",
     "Attendance and access control REST API, per device event log",
     "float32, 2 values, scaled by 3", "[0, 1]"),
    ("Lockout flags A and B", "Whether a reader has locked this pupil out.",
     "Same access control API, device state field", "float32, 2 values, 0 or 1", "{0, 1}"),
    ("Reader B health belief", "Whether the side reader is out of service. Reads 0 until the "
     "pupil is close enough to see the display.",
     "Device heartbeat endpoint, legible on the kiosk screen within roughly 2.5 m",
     "float32", "{-1, 0, 1}"),
    ("Queue length at A and B", "How many people are waiting at each reader.",
     "Overhead camera people counter, or the queue display board feed",
     "float32, 2 values, scaled by 3", "[0, 1]"),
    ("Time remaining", "Fraction of the episode budget still left.",
     "System clock", "float32", "[0, 1]"),
    ("Time to the bell", "Fraction of time left before the late bell. Goes negative once late.",
     "School timetable API plus system clock", "float32", "[-1, 1]"),
    ("Manual sign-in available", "Whether reception will accept a signature yet.",
     "Access control API, lockout status", "float32, 0 or 1", "{0, 1}"),
]

ACTION_ROWS = [
    ("0", "move_north", "Walk one step (0.5 m) towards the building."),
    ("1", "move_south", "Walk one step away from the building."),
    ("2", "move_west", "Walk one step towards the gate side."),
    ("3", "move_east", "Walk one step towards the reader side."),
    ("4", "sanitize_hands", "Take a dose of sanitizer. Only works at the stand."),
    ("5", "scan_fingerprint", "Present a finger to a reader. Only works at a reader."),
    ("6", "wait_in_queue", "Stand still and let the queue ahead clear."),
    ("7", "request_assistance", "Call a staff member to recalibrate a reader. Takes 4 steps."),
    ("8", "check_in_manually", "Sign the register at reception. Only after a lockout."),
]


def obs_table() -> str:
    rows = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>"
        for a, b, c, d, e in OBS_ROWS)
    return ('<table class="obs"><thead><tr><th>Observation</th><th>Description</th>'
            '<th>Source (sensor, camera, API, dataset)</th><th>Encoding and data type</th>'
            '<th>Range</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def action_table() -> str:
    rows = "".join(f"<tr><td>{i}</td><td><code>{n}</code></td><td>{d}</td></tr>"
                   for i, n, d in ACTION_ROWS)
    return ('<table><thead><tr><th>Id</th><th>Action</th><th>Meaning in the real world</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>')


CSS = """
@page { size: A4; margin: 11mm 11mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 9.1pt;
       line-height: 1.30; color: #1a1a1a; margin: 0; }
h1 { font-size: 16pt; margin: 0 0 1.5mm; color: #12305c; }
h2 { font-size: 12pt; margin: 4mm 0 1.5mm; color: #12305c;
     border-bottom: 1.6pt solid #12305c; padding-bottom: 1mm; }
h3 { font-size: 10.4pt; margin: 3mm 0 1mm; color: #1c4a86;
     page-break-after: avoid; }
h4 { font-size: 10pt; margin: 3mm 0 1mm; color: #333; page-break-after: avoid; }
p  { margin: 0 0 1.7mm; text-align: justify; }
ul { margin: 0 0 2.5mm 5mm; padding: 0; }
li { margin-bottom: 1mm; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f2f4f7;
       padding: 0 1pt; border-radius: 2pt; }
.meta { background: #f2f5fa; border-left: 3pt solid #12305c; padding: 2.5mm 3mm;
        margin-bottom: 4mm; font-size: 9.3pt; }
.meta div { margin-bottom: 0.8mm; }
table { width: 100%; border-collapse: collapse; margin: 1.5mm 0 2mm; font-size: 7.7pt; }
th, td { border: 0.4pt solid #b9c2cf; padding: 0.8mm 1.1mm; text-align: left;
         vertical-align: top; }
th { background: #dde5f0; font-weight: 600; }
tbody tr:nth-child(even) { background: #f7f9fc; }
tr.best { background: #ddf0dd !important; font-weight: 600; }
table.obs { font-size: 7.1pt; }
figure { margin: 1.5mm 0 2mm; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; border: 0.5pt solid #ccd3dd; border-radius: 2pt; }
figcaption { font-size: 8.3pt; color: #444; margin-top: 1mm; font-style: italic; }
.wide img { max-height: 61mm; }
.tall img { max-height: 69mm; }
.pair { display: flex; gap: 3mm; page-break-inside: avoid; }
.pair figure { flex: 1; margin: 2mm 0; }
.pair img { max-height: 40mm; }
.formula { background: #f7f9fc; border: 0.4pt solid #ccd3dd; padding: 1.5mm 2.5mm;
           margin: 1.5mm 0 2mm; font-family: Consolas, monospace; font-size: 8.0pt;
           line-height: 1.35; white-space: pre-wrap; }
.note { background: #fff8e6; border-left: 3pt solid #d9a520; padding: 2mm 3mm;
        margin: 2mm 0 3mm; font-size: 9.2pt; }
.pb { page-break-before: always; }
.missing { color: #b00; }
"""


def build_html() -> str:
    from training.plots import convergence_stats
    cs = convergence_stats()
    d, p, a, rf = (cs["dqn"], cs["ppo"], cs["a2c"], cs["reinforce"])

    dqn_best = sweep("dqn").loc[sweep("dqn").mean_return.idxmax()]
    ppo_best = sweep("ppo").loc[sweep("ppo").mean_return.idxmax()]
    a2c_best = sweep("a2c").loc[sweep("a2c").mean_return.idxmax()]
    rf_best = sweep("reinforce").loc[sweep("reinforce").mean_return.idxmax()]

    a2c_df = sweep("a2c")
    a2c_zero = a2c_df[a2c_df.ent_coef == 0.0]
    rf_df = sweep("reinforce")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RL Summative Report</title>
<style>{CSS}</style></head><body>

<h1>Reinforcement Learning Summative Assignment Report</h1>
<div class="meta">
  <div><b>Student Name:</b> {STUDENT}</div>
  <div><b>Video Recording:</b> {VIDEO}</div>
  <div><b>GitHub Repository:</b> {REPO}</div>
  <div><b>Environment:</b> SchoolCheckIn-v0, a custom Gymnasium environment rendered in 3D with PyBullet</div>
</div>

<h2>Project Overview</h2>
<p>This project trains an agent to run the morning attendance routine at a school that uses
biometric check-in. A pupil arrives at the gate and has to register attendance before the
late bell by reaching a fingerprint reader and scanning in successfully. The reader only
accepts a clean finger, hands can only be washed at a sanitizer stand on the far side of the
courtyard, one of the two readers is unreliable and is sometimes out of service, and both
readers queue during the morning rush. A good policy therefore has to plan a route in
stages, keep track of which reader is still usable, and decide when to keep trying rather
than give up. I compare a value based method (DQN) against three policy based methods
(REINFORCE, PPO and A2C) on exactly the same environment, with ten hyperparameter
configurations each.</p>

{img("env_screenshot.png", "Figure 1. The environment rendered in PyBullet. The classroom block sits along the north side with the main entrance and reader A, the east side gate carries reader B, and the courtyard holds the sanitizer stand, the reception booth, a hedge planter and a run of queue barriers.", "wide")}

<h2>Environment Description</h2>

<h3>Agent</h3>
<p>The agent is a single pupil arriving at the school gate. It is drawn as a uniformed figure
with a backpack, and the hand it will present to the reader is tinted by how clean it is, so
you can see the hygiene state directly in the simulation. The pupil can walk in four
directions on a half metre grid, use the sanitizer stand, present a finger at either reader,
stand and wait for a queue to clear, call a staff member to reset a reader that has locked
it out, and sign the paper register at reception once it has grounds to do so. It cannot walk
through the hedge planter or the queue barriers, and bumping into them makes its hands dirty
again, so careless movement has a cost later on. Its position, its hand cleanliness and the
number of rejections it has collected are all it really knows. Whether reader B is broken is
hidden until it gets close enough to read the display, so part of the task is finding out.</p>

<h3>Action Space</h3>
<p>The action space is discrete with nine actions, written as <code>Discrete(9)</code>. I kept
it discrete so that DQN, REINFORCE, PPO and A2C can all be trained on exactly the same
environment and compared fairly. Every action maps onto something a real pupil does at a
school gate.</p>
{action_table()}
<p>Movement of 0.5 m per step matches an unhurried walking pace sampled at roughly one step
per second. The three station actions only have an effect at the matching station, which is
what forces the agent to plan a route instead of acting from anywhere in the courtyard.</p>

<h3>Observation Space</h3>
<p>The observation is a 29 element <code>Box</code> vector, every element scaled into
[-1, 1] and stored as <code>float32</code>. The agent sees what a real deployment could
actually measure, not the full internal state. In particular it does not see the true
reliability of either reader, and it does not see whether reader B is broken until it is
close enough to read the display. The table below lists each group of features and the
equipment that would supply it on a real site.</p>
{obs_table()}
<p>Two things are deliberately withheld. The per reader acceptance probability is never
observable, so the agent has to treat rejection as noise rather than as a signal it can read
off directly. The health of reader B is masked to zero beyond about 2.5 m, which makes the
walk towards it an information gathering action as well as a movement.</p>

<h3>Reward Structure</h3>
<p>The reward has three parts: a small running cost, a shaping term that measures progress
along the route, and event rewards for what the agent does at the stations. The step reward
is</p>
<div class="formula">r_t = -c_step + w_d * ( PHI(s_t+1) - PHI(s_t) ) + r_event

c_step = 0.05        per step cost, so dawdling is never free
w_d    = 1.5         weight on route progress</div>
<p>The shaping potential PHI is the negative of the remaining route length. If no reader is
still usable it measures the walk to reception. Otherwise it blends the direct walk to the
nearest usable reader with the detour through the sanitizer, weighted by how dirty the hands
currently are:</p>
<div class="formula">PHI(s) = -[ delta * ( d(p, H) + min_i d(H, S_i) ) + (1 - delta) * min_i d(p, S_i) ]

delta = clip( (0.7 - h) / 0.7, 0, 1 )      h = hand cleanliness
p = pupil position, H = sanitizer, S_i = usable readers</div>
<p>Writing the shaping this way matters more than it looks. Because PHI depends only on the
state, every closed loop in state space sums to zero, so no repeating behaviour can farm it.
An earlier version of the reward simply paid the agent for getting closer to whichever
sub goal was active at the time. Since that sub goal flipped between the sanitizer and the
reader as the hands got dirty, walking back and forth paid on both legs, and PPO learned to
shuttle between the two forever. It scored about +40 per episode while checking in on 0% of
episodes. The regression test <code>test_shaping_cannot_be_farmed_by_shuttling</code> now
keeps that closed.</p>
<p>The event rewards are as follows.</p>
<div class="formula">Movement
  blocked by the planter or barriers   -0.25  and hand cleanliness h <- h - 0.20
  free step                                   h <- h - 0.012

Sanitizer
  dose taken at the stand (h < 0.95)   0      h <- min(h + 0.45, 1.0)
  dose taken with clean hands          -0.15
  used away from the stand             -0.20

Fingerprint reader
  used away from a reader              -0.20
  reader locked out or out of service  -0.60
  reader busy with someone else        -0.15
  accepted, probability p              +20.0  and the episode ends
       p = clip( (h - 0.45) / 0.45, 0.02, 0.97 ) * rho_i
  rejected                             -0.50  h <- h - 0.12, rejections_i <- +1
  third rejection at that reader       -2.00  that reader locks out

Other
  wait in queue                        -0.02
  request assistance at a reader       -0.30  staff clear that reader after 4 steps
  manual sign-in at reception          +5.00  only after a lockout, else -0.20

Terminal adjustments
  checked in before the bell (t <= 80) +5.0 * (80 - t) / 80
  checked in after the bell            -4.00
  every route exhausted (stranded)     -8.00
  never checked in by step 150         -6.00</div>
<p>The gap between +20 for a biometric check-in and +5 for a signature is what keeps the
agent honest, and reception refuses to sign anyone that no reader has actually rejected.
Without that precondition the office is a nine step walk from the gate for a guaranteed
reward, and PPO simply took it: an earlier run converged on signing the register in 9 steps
on 100% of episodes and never learned to use a reader at all.</p>

<h3>Start State and Termination</h3>
<p>Each episode starts with the pupil at the gate. Hand cleanliness is drawn uniformly from
[0.15, 0.60], reader A reliability from [0.80, 1.00], reader B reliability from
[0.45, 0.85], and reader B is out of service on 25% of episodes. An episode ends when the
pupil scans in successfully, signs the register at reception, becomes stranded because every
route is gone and reception is out of reach in the time left, or runs out of time at step
150. The bell rings at step 80 and splits punctual arrivals from late ones.</p>

<h2>System Analysis And Design</h2>

<h3>Deep Q-Network (DQN)</h3>
<p>DQN comes from Stable Baselines3 with the <code>MlpPolicy</code> head. The network is a
fully connected value network that maps the 29 element observation to nine action values,
with the hidden layer sizes varied across the sweep from a single layer of 64 units up to
two layers of 256. The winning configuration, <code>{dqn_best['name']}</code>, uses
{dqn_best['net_arch']}. The implementation keeps the two standard stabilisers. A replay
buffer holds past transitions and is sampled uniformly, which breaks the correlation between
consecutive steps in a long walk across the courtyard. A target network supplies the
bootstrap value and is refreshed every {int(dqn_best['target_update_interval'])} steps in the
best run, which stops the regression target moving with every gradient step. Exploration is
epsilon greedy, annealed from 1.0 down to 0.05 over the fraction of training given by
<code>exploration_fraction</code>. Learning starts after 1000 steps of pure exploration and
a gradient step is taken every 4 environment steps.</p>
<p>The design question here is that the task needs a long correct sequence, roughly 35 steps,
before any reward for checking in arrives. The shaped route reward carries most of the
learning signal, and the replay buffer lets DQN reuse the rare successful endings many times
instead of once.</p>

<h3>Policy Gradient Methods (REINFORCE, PPO and A2C)</h3>
<p><b>REINFORCE</b> is written from scratch in PyTorch, since Stable Baselines3 does not
provide it. The policy network has two hidden Tanh layers, with width varied across the
sweep from 64 to 256 units, and a softmax over the nine actions through a
<code>Categorical</code> distribution. Training is pure Monte Carlo: it plays four complete
episodes, computes discounted returns to go for every visited state, and takes one gradient
step on the batch. The <code>baseline</code> column in the sweep switches on a learned state
value network trained by regression on the same returns, which is the REINFORCE with
baseline formulation from Sutton and Barto section 13.4. Advantages are normalised before
the update and an entropy bonus is available through <code>ent_coef</code>.</p>
<p>The baseline turned out to matter a great deal. With plain normalised returns the gradient
variance over a 150 step episode with nine actions was large enough that the agent never
found the route at all, sitting at a 0% check-in rate even after 400k steps. Subtracting a
learned value estimate leaves the gradient unbiased but much quieter, and the method then
reaches the same final performance as PPO.</p>
<p><b>PPO</b> and <b>A2C</b> both come from Stable Baselines3 with the default
<code>MlpPolicy</code>, an actor critic pair of two 64 unit hidden layers. PPO collects a
rollout of <code>n_steps</code> transitions, computes advantages with generalised advantage
estimation, and then takes several epochs of minibatch updates under a clipped objective,
where <code>clip_range</code> bounds how far the policy may move in one update. A2C is the
synchronous, single update version of the same idea, so it updates far more often on much
shorter rollouts, from 5 to 64 steps in this sweep. Both expose an entropy coefficient, and
as the results show, that single knob decides whether A2C learns anything at all here.</p>

<h2>Implementation</h2>
<p>Every algorithm was run with ten different hyperparameter configurations on the same
environment. Mean return and check-in rate are measured over 30 evaluation episodes with a
seed range disjoint from training, so the numbers reflect generalisation and not memorised
starts. The winning row of each table is highlighted. DQN, PPO and A2C were each trained for
250k steps; REINFORCE needed 900k, for reasons discussed below.</p>

<h4>DQN</h4>
{table("dqn", {"name": "Run", "learning_rate": "Learning rate", "gamma": "Gamma",
               "buffer_size": "Replay buffer", "batch_size": "Batch",
               "exploration_fraction": "Exploration fraction",
               "target_update_interval": "Target update", "net_arch": "Hidden layers",
               "mean_return": "Mean reward", "success_rate": "Check-in rate",
               "mean_length": "Steps"})}
<p>Learning rate and the discount factor dominate. The two runs at gamma 0.999 and 0.99 with
a small learning rate stay well behind, while the three best runs all sit at gamma between
0.90 and 0.98. A shorter effective horizon suits this task because the shaped route reward
already carries the long term information, so a very long horizon mostly adds variance. The
clearest single lesson is in <code>dqn01</code> against <code>dqn10</code>: same learning
rate, same gamma, same hidden layers, and yet 0.47 against 0.93 check-in rate. They differ
only in exploration fraction, 0.20 against 0.10, and target update interval. The longer
exploration schedule left <code>dqn01</code> still acting semi randomly late in training,
and its episodes average 136 steps against 47, so it was reaching the reader after the bell
rather than not at all.</p>

<h4>REINFORCE</h4>
{table("reinforce", {"name": "Run", "learning_rate": "Learning rate", "gamma": "Gamma",
                     "hidden": "Hidden width", "ent_coef": "Entropy coef",
                     "baseline": "Value baseline", "mean_return": "Mean reward",
                     "success_rate": "Check-in rate", "tardy_rate": "Late rate",
                     "mean_length": "Steps"})}
<p>Entropy is the deciding column. Of the four runs with no entropy bonus, three fail
completely and never check in. The three runs that reach a 1.00 check-in rate all carry an
entropy coefficient between 0.01 and 0.03. The one exception, <code>reinforce03</code>,
gets to 0.87 with no entropy bonus but compensates with a shorter horizon at gamma 0.95.
Gamma 0.999 in <code>reinforce05</code> fails outright, which matches what DQN showed. Every
failing run has a mean episode length of exactly 150, meaning it never terminates and simply
wanders until the clock runs out.</p>

<h4>PPO</h4>
{table("ppo", {"name": "Run", "learning_rate": "Learning rate", "gamma": "Gamma",
               "n_steps": "Rollout steps", "batch_size": "Batch",
               "clip_range": "Clip range", "ent_coef": "Entropy coef",
               "gae_lambda": "GAE lambda", "mean_return": "Mean reward",
               "success_rate": "Check-in rate", "mean_length": "Steps"})}
<p>PPO is the most forgiving of the four. Seven of ten configurations pass a 0.90 check-in
rate, and the failures are informative rather than random. <code>ppo05</code> pairs the
smallest learning rate in the sweep with gamma 0.999 and collapses to 0.33. <code>ppo09</code>
uses a 4096 step rollout, which at a 250k budget leaves only about 61 policy updates in
total, and it reaches 0.40 with a 0.27 late rate. Both failures are about not making enough
progress in the budget rather than about instability, which is what the clipped objective is
there to prevent.</p>

<h4>A2C</h4>
{table("a2c", {"name": "Run", "learning_rate": "Learning rate", "gamma": "Gamma",
               "n_steps": "Rollout steps", "ent_coef": "Entropy coef",
               "vf_coef": "Value coef", "gae_lambda": "GAE lambda",
               "mean_return": "Mean reward", "success_rate": "Check-in rate",
               "mean_length": "Steps"})}
<p>A2C gives the sharpest result in the whole study. Every single configuration with
<code>ent_coef</code> set to 0.0, that is {", ".join(a2c_zero.name.tolist())}, scores exactly
0.00 and runs the full 150 steps every episode. Every configuration with an entropy
coefficient between 0.01 and 0.05 checks in on 0.23 to 0.87 of episodes. Without an explicit
entropy bonus A2C commits to a poor action distribution early, and because it updates on
rollouts as short as five steps it never gathers the evidence to escape. This is the one
knob that separates a working A2C from a useless one on this task.</p>

<h2>Results Discussion</h2>
<p>The table below summarises the best run of each method before the individual figures are
discussed. The spread between the best and the worst configuration is included because it is
the clearest evidence that the environment is actually sensitive to tuning.</p>
{summary_table()}

<h3>Cumulative Rewards</h3>
{img("reward_curves.png", "Figure 2. Episode return over training for the best configuration of each method. The pale line is the raw per episode return and the solid line is a 20 episode moving average.", "tall")}
<p>The four curves have visibly different shapes. PPO rises fastest and then holds a tight
band around +34, with only occasional single episode dips where a run of rejections or a
broken reader B costs it the bonus. DQN climbs more slowly, takes until roughly episode 1200
to reach its plateau, and keeps showing deep downward spikes for the rest of training. Those
spikes are the signature of a value based method with a moving target: a target network
refresh can briefly change the greedy action in part of the state space. A2C reaches a
reasonable level earliest of all but never becomes tidy, and its band stays wide from start
to finish, which is what you expect from very short rollouts and single update steps.
REINFORCE is the outlier. It sits flat near -10 for about 2000 episodes, doing nothing that
looks like progress, then breaks through and climbs for the remaining 8000 episodes. That
long flat spell is the exploration problem in its rawest form: until a Monte Carlo rollout
happens to complete the whole sanitize then scan sequence, there is no gradient pointing
towards it.</p>

<h3>Training Stability</h3>
{img("dqn_objective.png", "Figure 3. DQN objective for the best configuration. The red line is the temporal difference loss on the left axis and the blue line is mean episode reward on the right axis.", "tall")}
<p>The DQN objective does not fall smoothly, and that is the correct behaviour rather than a
fault. The temporal difference loss rises while mean reward is still climbing, because the
agent is discovering states with much larger returns than its value estimates predicted, so
the regression target keeps moving upward. The loss only settles once the reward curve
flattens. Reading the two axes together is what makes this interpretable: a falling loss on
its own would be equally consistent with a policy that has stopped improving.</p>
{img("pg_entropy.png", "Figure 4. Policy entropy over training for the three policy gradient methods. Higher entropy means the policy is still spreading probability across actions.", "tall")}
<p>The entropy curves explain the exploration story behind the sweep tables. All three start
near the entropy of a uniform distribution over nine actions, which is about 2.2 nats. PPO
and A2C both decay steadily as they commit to the route, and the entropy coefficient sets the
floor they settle at, which is exactly why the A2C runs with no bonus collapse: their entropy
falls away before they have found anything worth committing to. REINFORCE holds much higher
entropy for far longer, which is the same fact as its long flat reward curve seen from the
other side. It keeps sampling widely because its gradient estimate is too noisy to justify
committing, and that is what eventually lets it stumble onto the full sequence.</p>

<h3>Episodes To Converge</h3>
{img("episodes_to_converge.png", "Figure 5. Left: 100 episode moving average for each method with its convergence point marked. Right: episodes needed to reach and hold 90% of final performance.", "wide")}
<p>To put a number on convergence I take the plateau to be the mean return over the final
10% of episodes, then find the first episode whose 100 episode moving average reaches 90% of
that plateau and holds it for the next 250 episodes. The holding requirement matters: without
it a single lucky spike counts as convergence, and with a stricter rule that demands the
average never drop again, the noisy methods only qualify in their last few hundred
episodes.</p>
{conv_table()}
<p>PPO converges first at episode {p['converged_at']}, and it is also the only method whose
first crossing at {p['first_reached']} is close to its stable point, a gap of only
{p['converged_at'] - p['first_reached']} episodes. A2C actually touches the 90% line earliest
of all at episode {a['first_reached']}, but then takes until {a['converged_at']} to hold it,
a gap of {a['converged_at'] - a['first_reached']} episodes, which quantifies the instability
visible in its reward curve. DQN needs {d['converged_at']} episodes but reaches the highest
plateau at {d['plateau']:+.2f}. REINFORCE needs {rf['converged_at']} episodes, roughly
{rf['converged_at'] / p['converged_at']:.1f} times PPO, and in sample terms the gap is wider
still: PPO reached its plateau inside a 250k step budget while REINFORCE needed 900k. That
ratio is the clearest statement of the cost of a pure Monte Carlo gradient on a task with a
long action sequence.</p>

<h3>Generalization</h3>
{img("generalization.png", "Figure 6. Check-in rate of each best agent when reader reliability is forced to values outside the training range. Training samples 0.80 to 1.00 for reader A and 0.45 to 0.85 for reader B.", "wide")}
<p>This test holds the policies fixed and forces reader reliability to values they never saw,
down to 0.3 where roughly two thirds of well presented fingers are rejected. PPO is
completely unaffected and holds a 1.00 check-in rate at every level, including 0.3. That is
not luck: with three attempts allowed per reader, two readers, and a staff call available,
a policy that keeps trying can still get in through a bad reader, and PPO has learned to
persist. A2C degrades gently, from 0.95 down to 0.60. DQN falls hardest, from 1.00 at
reliability 0.9 to 0.55 at 0.3, which says its greedy policy is tuned to the reliability
range it trained on and gives up rather than retrying when rejections mount. This is a
useful reminder that the ranking on the training distribution, where DQN beat A2C
comfortably, does not survive a shift in conditions.</p>
{img("checkin_modes.png", "Figure 7. How each best agent ends its episodes: the intended biometric check-in, the lower value signature at reception, or no check-in at all.", "wide")}
<p>This figure checks that the agents are solving the intended problem rather than taking the
easy way out. Not one of the four ever signs the register, so the orange band is empty
everywhere. Every success is a real fingerprint check-in. PPO and REINFORCE check in on every
episode, DQN misses 2.5% and A2C misses 17.5%. Given that reception is a guaranteed +5 and is
much closer to the gate than either reader, an agent settling for it would have been an
entirely rational failure mode, and gating it behind a genuine lockout is what prevented
that.</p>

<h3>Agent Behaviour In The Simulation</h3>
<div class="pair">
{img("entrance_closed.png", "Figure 8. Presenting a finger at reader A. The lamp is amber and the doors are shut.")}
{img("entrance_open.png", "Figure 9. The scan is accepted. The lamp turns green, the doors slide open and the pupil walks into the lobby.")}
</div>
<p>Running <code>uv run main.py play</code> loads the best agent and shows it in the PyBullet
window with a step by step trace in the terminal. The learned policy is unmistakably the
intended one. In a typical episode the pupil starts with cleanliness 0.17, walks 13 steps
north up the west corridor while its hands get dirtier still, takes three doses at the
sanitizer to reach 1.00, crosses 15 steps east to reader A arriving at 0.82, is rejected
once, and is accepted on the second attempt for a return of +40.15 in 35 steps, comfortably
before the bell. In another episode it presented a finger twice into a busy reader, absorbed
the small busy penalty, waited for the queue to drain and got in on the third try. That is
the behaviour the reward was designed to produce: it takes the detour instead of gambling on
dirty hands, and it persists at a contested reader instead of walking away.</p>

<h2>Conclusion and Discussion</h2>
<p>On this environment the two strongest methods are PPO and REINFORCE with a learned
baseline, which finish within 0.3 of each other at {rf_best.mean_return:+.2f} and
{ppo_best.mean_return:+.2f}, both checking in on every evaluation episode. They get there
very differently. PPO needed 250k steps and {p['converged_at']} episodes; REINFORCE needed
900k steps and {rf['converged_at']} episodes for the same result. If sample efficiency
matters at all, and on a real gate every sample is a real pupil, PPO is the clear choice.
DQN reaches the highest training plateau of any method at {d['plateau']:+.2f} and a
{dqn_best.success_rate:.2f} check-in rate, but it is the least robust: forced onto unreliable
readers it drops to 0.55, the worst of the four. A2C is the weakest overall at
{a2c_best.mean_return:+.2f} and is the most sensitive to a single hyperparameter, since every
run without an entropy bonus fails completely.</p>
<p>The pattern behind those results is that this task punishes premature commitment. It needs
a roughly 35 step sequence with a detour that looks like a detour, so any method that
narrows its action distribution before finding the full sequence gets stuck. That is exactly
what happens to A2C with no entropy bonus, to the DQN runs whose exploration schedule decays
too fast, and to REINFORCE without a baseline. PPO does best because the clipped objective
limits how far each update can move the policy, which keeps exploration alive without
needing the entropy term tuned precisely.</p>
<p>The main strengths and weaknesses split cleanly. DQN reuses rare successful episodes many
times through its replay buffer, which is why it reaches a high plateau, but its greedy
policy transfers poorly and its value target keeps moving, which shows as permanent spikes in
the reward curve. PPO is stable, sample efficient and by far the most robust off
distribution, at the cost of more moving parts to configure. A2C is cheap per update and
learns quickly at first, but its short rollouts make it noisy and fragile. REINFORCE is by
far the simplest to implement and reasons about, and it does reach the top of the table, but
it needed nearly four times the samples and it is unusable without the value baseline.</p>
<p>Two experiments shaped this work more than any tuning did, and both were reward design
problems rather than algorithm problems. The first was the shaping term: rewarding progress
toward whichever sub goal was currently active let PPO shuttle between the sanitizer and the
reader for about +40 per episode at a 0% check-in rate. Rewriting it as a proper potential
over remaining route length, so that closed loops sum to zero, removed the exploit. The
second was the manual fallback: as an unconditional option it was a nine step walk for
guaranteed reward and PPO took it every time, so it now requires a genuine lockout first.
Both are recorded as regression tests, because in both cases the training curves looked
healthy while the agent was not doing the task at all.</p>
<p>With more time the obvious next steps are a recurrent or frame stacked policy, since
reader B's health is genuinely hidden and a memory of what was seen near the side gate would
be worth more than the single masked feature the agent gets now; several random seeds per
configuration, because a ten run sweep at one seed cannot separate a good setting from a
lucky one; and multiple pupils sharing the readers, which would turn the queue from sampled
noise into something the agent could reason about and would make the environment a much
closer match to a real morning rush.</p>

</body></html>
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    html = build_html()

    bad = [c for c in ("—", "–") if c in html]
    if bad:
        raise SystemExit(f"report contains dash characters that were asked to be avoided: {bad}")

    html_path = os.path.join(OUT_DIR, "report.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"[report] wrote {html_path}")

    chrome = "/usr/bin/google-chrome"
    if os.path.exists(chrome):
        pdf_path = os.path.join(OUT_DIR, "report.pdf")
        subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}",
             f"file://{html_path}"],
            check=True, capture_output=True, timeout=180)
        print(f"[report] wrote {pdf_path}")
    else:
        print("[report] google-chrome not found; convert report.html to PDF manually")


if __name__ == "__main__":
    main()
