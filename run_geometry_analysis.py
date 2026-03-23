"""
run_geometry_analysis.py
Batch version of geometry_compare.ipynb — runs all hypothesis tests and saves
plots to data/geometry_analysis_plots/. Designed to be submitted via sbatch.

Usage:
    python run_geometry_analysis.py [--pkl data/fur_anchor_embeddings.pkl]
                                    [--pkl30 data/fur_anchor_embeddings_30.pkl]
                                    [--bc30 data/best_conditions_30.json]
                                    [--outdir data/geometry_analysis_plots]
"""

import argparse, os, pickle, json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')   # no display server needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--pkl',    default='data/fur_anchor_embeddings.pkl',
                    help='Pilot (6q) anchor embeddings pkl')
parser.add_argument('--pkl30',  default='data/fur_anchor_embeddings_30.pkl',
                    help='30-question anchor embeddings pkl (if available)')
parser.add_argument('--bc30',   default='data/best_conditions_30.json',
                    help='best_conditions JSON for 30q run (if available)')
parser.add_argument('--outdir', default='data/geometry_analysis_plots',
                    help='Directory for output plots and tables')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ─── Shared constants ─────────────────────────────────────────────────────────
TRAJ_KEYS = [
    'Answer_A', 'Reasoning_A', 'Answer_B', 'Reasoning_B',
    'Answer_C', 'Reasoning_C', 'Answer_D', 'Reasoning_D', 'Final_Answer',
]
BLOCK_LETTERS = ['A', 'B', 'C', 'D']
STEP_TO_BLOCK = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'Final'}

def reasoning_delta(rec, letter):
    ae = rec['anchor_embeddings']
    a = ae.get(f'Answer_{letter}')
    r = ae.get(f'Reasoning_{letter}')
    if a is None or r is None:
        return None
    return float(np.linalg.norm(r.astype(np.float32) - a.astype(np.float32)))

def cosine_sim(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ─── Geometric feature functions (from Cell 1) ────────────────────────────────
def pts(df): return df[['pc1', 'pc2', 'pc3']].values
def arclength(df):
    p = pts(df); return np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
def net_displacement(df):
    p = pts(df); return np.linalg.norm(p[-1] - p[0])
def straightness(df):
    a, n = arclength(df), net_displacement(df); return n / a if a > 1e-9 else 0.
def mean_step_size(df):
    p = pts(df); return np.linalg.norm(np.diff(p, axis=0), axis=1).mean()
def step_size_std(df):
    p = pts(df); return np.linalg.norm(np.diff(p, axis=0), axis=1).std()
def mean_curvature(df):
    p = pts(df)
    if len(p) < 3: return 0.
    d1, d2 = np.diff(p, axis=0), np.diff(np.diff(p, axis=0), axis=0)
    norms = np.linalg.norm(d1[:-1], axis=1)
    cross = np.linalg.norm(np.cross(d1[:-1], d2), axis=1)
    mask = norms > 1e-9
    return float(np.mean(cross[mask] / (norms[mask] ** 2 + 1e-9))) if mask.any() else 0.
def final_pc1(df): return float(pts(df)[-1, 0])
def final_pc2(df): return float(pts(df)[-1, 1])
def final_pc3(df): return float(pts(df)[-1, 2])
def pc1_range(df): return float(pts(df)[:, 0].max() - pts(df)[:, 0].min())
def pc2_range(df): return float(pts(df)[:, 1].max() - pts(df)[:, 1].min())

METRICS = [
    ('arclength',        arclength),
    ('net_displacement', net_displacement),
    ('straightness',     straightness),
    ('mean_step_size',   mean_step_size),
    ('step_size_std',    step_size_std),
    ('mean_curvature',   mean_curvature),
    ('final_pc1',        final_pc1),
    ('final_pc2',        final_pc2),
    ('final_pc3',        final_pc3),
    ('pc1_range',        pc1_range),
    ('pc2_range',        pc2_range),
]

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Load pilot pkl, fit shared PCA, plot trajectories
# ══════════════════════════════════════════════════════════════════════════════
print('=' * 80)
print('CELL 3 — Shared PCA trajectories (pilot 6q pkl)')
print('=' * 80)

with open(args.pkl, 'rb') as f:
    records = pickle.load(f)

print(f'Loaded {len(records)} records from {args.pkl}')
for r in sorted(records, key=lambda x: x['ff_soft']):
    cov = sum(1 for v in r['anchor_embeddings'].values() if v is not None)
    print(f"  {r['question_id']:20s}  ff_soft={r['ff_soft']:.4f}  "
          f"condition={r['condition']}  epoch={r['epoch']}  anchors={cov}/9")

all_vecs = []
for rec in records:
    for k in TRAJ_KEYS:
        v = rec['anchor_embeddings'].get(k)
        if v is not None:
            all_vecs.append(v)
X_all = np.stack(all_vecs).astype(np.float32)
pca = PCA(n_components=3, random_state=42).fit(X_all)
ev = pca.explained_variance_ratio_
print(f'\nShared PCA explained variance:  PC1={ev[0]:.3f}  PC2={ev[1]:.3f}  PC3={ev[2]:.3f}  '
      f'(cumulative={ev[:3].sum():.3f})')

trajs = []
for rec in records:
    vecs = [rec['anchor_embeddings'].get(k) for k in TRAJ_KEYS]
    valid_keys  = [k for k, v in zip(TRAJ_KEYS, vecs) if v is not None]
    valid_vecs  = [v for v in vecs if v is not None]
    if len(valid_vecs) < 2:
        print(f"  [SKIP] {rec['question_id']} — too few anchors")
        continue
    proj = pca.transform(np.stack(valid_vecs).astype(np.float32))
    df_t = pd.DataFrame(proj, columns=['pc1', 'pc2', 'pc3'])
    df_t['anchor'] = valid_keys
    trajs.append({'df': df_t, 'id': rec['question_id'],
                  'ff_soft': rec['ff_soft'], 'condition': rec['condition'],
                  'epoch': rec['epoch']})
trajs.sort(key=lambda x: x['ff_soft'])

cmap   = plt.cm.RdYlBu
colors = [cmap(t['ff_soft']) for t in trajs]

fig = plt.figure(figsize=(15, 6))
ax3d = fig.add_subplot(121, projection='3d')
for t, c in zip(trajs, colors):
    p = t['df'][['pc1', 'pc2', 'pc3']].values
    ax3d.plot(p[:, 0], p[:, 1], p[:, 2], '-o', color=c, lw=2, markersize=5,
              label=f"{t['id']}  (ff={t['ff_soft']:.3f})", alpha=0.9)
    ax3d.scatter(*p[0],  s=80, c='black', marker='s', zorder=5)
    ax3d.scatter(*p[-1], s=80, c=[c],     marker='*', zorder=5)
    for i, row in enumerate(p):
        ax3d.text(row[0], row[1], row[2], str(i + 1), fontsize=7)
ax3d.set_xlabel('PC1'); ax3d.set_ylabel('PC2'); ax3d.set_zlabel('PC3')
ax3d.set_title('3D Anchor Trajectories  (■=start  ★=end)')
ax3d.legend(fontsize=7, loc='upper left')

ax2d = fig.add_subplot(122)
for t, c in zip(trajs, colors):
    p = t['df'][['pc1', 'pc2', 'pc3']].values
    ax2d.plot(p[:, 0], p[:, 1], '-o', color=c, lw=2, markersize=7,
              label=f"{t['id']}  (ff={t['ff_soft']:.3f})", alpha=0.9)
    for i, (x, y) in enumerate(zip(p[:, 0], p[:, 1])):
        ax2d.annotate(str(i + 1), (x, y), fontsize=8, ha='center', va='bottom', color=c)
ax2d.set_xlabel('PC1'); ax2d.set_ylabel('PC2')
ax2d.set_title('PC1 vs PC2  (step numbers labelled)')
ax2d.legend(fontsize=7)

plt.suptitle(
    f'Best-Conditions MCQ — Shared PCA Trajectories  (n={len(trajs)})\n'
    f'PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  PC3={ev[2]:.1%}  cumulative={ev[:3].sum():.1%}',
    fontsize=12,
)
plt.tight_layout()
plot_path = os.path.join(args.outdir, 'pca_trajectories.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {plot_path}')

rows_traj = []
for t in trajs:
    row = {'id': t['id'], 'ff_soft': round(t['ff_soft'], 4),
           'condition': t['condition'], 'epoch': t['epoch']}
    for name, fn in METRICS:
        row[name] = round(fn(t['df']), 4)
    rows_traj.append(row)
print('\nPer-trajectory metrics:')
print(pd.DataFrame(rows_traj).set_index('id').to_string())

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Null baseline
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 80)
print('CELL 5 — Null baseline (1000 random anchor pairs)')
print('=' * 80)

anchor_pool = []
for rec in records:
    for k in TRAJ_KEYS:
        v = rec['anchor_embeddings'].get(k)
        if v is not None:
            anchor_pool.append(v.astype(np.float32))
anchor_pool = np.stack(anchor_pool)
print(f'Anchor pool: {anchor_pool.shape[0]} vectors of dim {anchor_pool.shape[1]}')

rng = np.random.default_rng(42)
rand_cosines, rand_deltas = [], []
for _ in range(1000):
    i, j = rng.choice(len(anchor_pool), size=2, replace=False)
    a, b = anchor_pool[i], anchor_pool[j]
    rand_cosines.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    rand_deltas.append(float(np.linalg.norm(a - b)))
rand_cosines = np.array(rand_cosines)
rand_deltas  = np.array(rand_deltas)

print(f'  Random cosine:  mean={rand_cosines.mean():.4f}  std={rand_cosines.std():.4f}')
print(f'  Random delta:   mean={rand_deltas.mean():.4f}   std={rand_deltas.std():.4f}')
print(f'  Cosine 1-sigma: [{rand_cosines.mean()-rand_cosines.std():.4f},  {rand_cosines.mean()+rand_cosines.std():.4f}]')
print(f'  Delta  1-sigma: [{rand_deltas.mean()-rand_deltas.std():.4f},  {rand_deltas.mean()+rand_deltas.std():.4f}]')

# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Test 1: reasoning_delta magnitudes & intra-CoT variance
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 80)
print('CELL 6 — TEST 1: reasoning_delta magnitudes and intra-CoT variance')
print('=' * 80)

rows_t1 = []
for rec in sorted(records, key=lambda x: x['ff_soft']):
    ae  = rec['anchor_embeddings']
    qid = rec['question_id']
    ff  = rec['ff_soft']
    deltas = {}
    for letter in BLOCK_LETTERS:
        a = ae.get(f'Answer_{letter}')
        r = ae.get(f'Reasoning_{letter}')
        deltas[letter] = float(np.linalg.norm(
            r.astype(np.float32) - a.astype(np.float32)
        )) if a is not None and r is not None else None

    valid_deltas = [v for v in deltas.values() if v is not None]
    intra_var = float(np.var(valid_deltas)) if len(valid_deltas) >= 2 else None

    fa_vec = ae.get('Final_Answer')
    rd_vec = ae.get('Reasoning_D')
    final_delta = float(np.linalg.norm(
        fa_vec.astype(np.float32) - rd_vec.astype(np.float32)
    )) if fa_vec is not None and rd_vec is not None else None

    rows_t1.append({
        'question_id':    qid,
        'ff_soft':        round(ff, 4),
        'delta_A':        round(deltas['A'], 4) if deltas['A'] is not None else None,
        'delta_B':        round(deltas['B'], 4) if deltas['B'] is not None else None,
        'delta_C':        round(deltas['C'], 4) if deltas['C'] is not None else None,
        'delta_D':        round(deltas['D'], 4) if deltas['D'] is not None else None,
        'intra_variance': round(intra_var, 6)   if intra_var   is not None else None,
        'final_delta':    round(final_delta, 4) if final_delta  is not None else None,
    })

print(pd.DataFrame(rows_t1).set_index('question_id').to_string())

ff_vals = [r['ff_soft']        for r in rows_t1 if r['intra_variance'] is not None]
iv_vals = [r['intra_variance'] for r in rows_t1 if r['intra_variance'] is not None]
ff_fd   = [r['ff_soft']        for r in rows_t1 if r['final_delta']    is not None]
fd_vals = [r['final_delta']    for r in rows_t1 if r['final_delta']    is not None]
r_iv, p_iv = stats.spearmanr(ff_vals, iv_vals)
r_fd, p_fd = stats.spearmanr(ff_fd,   fd_vals)

print(f'\n  ff_soft vs intra_variance:  r={r_iv:.3f}, p={p_iv:.3f} (n={len(ff_vals)})')
print(f'  ff_soft vs final_delta:     r={r_fd:.3f}, p={p_fd:.3f} (n={len(ff_fd)})')
all_deltas_flat = [r[k] for r in rows_t1 for k in ['delta_A','delta_B','delta_C','delta_D'] if r[k] is not None]
within = sum(1 for d in all_deltas_flat if rand_deltas.mean()-rand_deltas.std() < d < rand_deltas.mean()+rand_deltas.std())
print(f'  Observed deltas within null band: {within}/{len(all_deltas_flat)}')

# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Test 2: ff_soft vs reasoning_delta (target block A + controls)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 80)
print('CELL 7 — TEST 2: ff_soft vs reasoning_delta (pilot, step_idx=0)')
print('=' * 80)

ff_list, da_list, db_list, dc_list, dd_list = [], [], [], [], []
for rec in records:
    ae  = rec['anchor_embeddings']
    row_d = {}
    for letter in BLOCK_LETTERS:
        a = ae.get(f'Answer_{letter}')
        r = ae.get(f'Reasoning_{letter}')
        row_d[letter] = float(np.linalg.norm(
            r.astype(np.float32) - a.astype(np.float32)
        )) if a is not None and r is not None else None
    if all(row_d[l] is not None for l in BLOCK_LETTERS):
        ff_list.append(rec['ff_soft'])
        da_list.append(row_d['A']); db_list.append(row_d['B'])
        dc_list.append(row_d['C']); dd_list.append(row_d['D'])

n = len(ff_list)
r_A, p_A = stats.spearmanr(ff_list, da_list)
r_B, p_B = stats.spearmanr(ff_list, db_list)
r_C, p_C = stats.spearmanr(ff_list, dc_list)
r_D, p_D = stats.spearmanr(ff_list, dd_list)

print(f'  ff_soft vs delta_A (TARGET): r={r_A:.3f}, p={p_A:.3f} (n={n})')
for lbl, r_v, p_v in [('B', r_B, p_B), ('C', r_C, p_C), ('D', r_D, p_D)]:
    print(f'  ff_soft vs delta_{lbl} (ctrl):   r={r_v:.3f}, p={p_v:.3f}')
r_vals = {'A': r_A, 'B': r_B, 'C': r_C, 'D': r_D}
strongest = max(r_vals, key=lambda k: abs(r_vals[k]))
if strongest == 'A':
    print('  -> Answer A shows strongest |r| — consistent with hypothesis.')
else:
    print(f'  -> Answer {strongest} shows stronger |r| than A ({abs(r_vals[strongest]):.3f} vs {abs(r_A):.3f}).')

# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Test 3: Conclusion-probe cosine coupling (pilot)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 80)
print('CELL 8 — TEST 3: Conclusion-probe cosine coupling (pilot)')
print('=' * 80)

usable = [rec for rec in records
          if any(v is not None for v in rec.get('conclusion_embeddings', {}).values())]
if not usable:
    print('No conclusion embeddings found — Test 3 deferred to 30q run.')
else:
    rows_t3 = []
    for rec in sorted(usable, key=lambda x: x['ff_soft']):
        ae = rec['anchor_embeddings']
        ce = rec['conclusion_embeddings']
        row = {'question_id': rec['question_id'], 'ff_soft': round(rec['ff_soft'], 4)}
        for letter in BLOCK_LETTERS:
            reas = ae.get(f'Reasoning_{letter}')
            conc = ce.get(letter)
            row[f'cos_{letter}'] = round(cosine_sim(reas, conc), 4) if reas is not None and conc is not None else None
        rows_t3.append(row)
    print(pd.DataFrame(rows_t3).set_index('question_id').to_string())
    print(f'Null cosine band: [{rand_cosines.mean()-rand_cosines.std():.4f}, {rand_cosines.mean()+rand_cosines.std():.4f}]')

# ══════════════════════════════════════════════════════════════════════════════
# CELLS 10–13 — Part 3: stronger tests using best_conditions + step_idx
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 80)
print('PART 3 — Stronger Test 2 + Full Test 3 (30q if available, else pilot)')
print('=' * 80)

# ── Cell 10: data loader ──────────────────────────────────────────────────────
if os.path.exists(args.pkl30) and os.path.exists(args.bc30):
    PKL_PATH, BC_PATH, DATA_TAG = args.pkl30, args.bc30, '30-question'
else:
    PKL_PATH, BC_PATH, DATA_TAG = args.pkl, None, '6-question (pilot, UNDERPOWERED)'
    print(f'[INFO] 30-question data not found — using pilot data.')
    print('[INFO] This is a sanity check only — n=6, step_idx=0 for all questions.')

with open(PKL_PATH, 'rb') as f:
    records_p3 = pickle.load(f)
print(f'[{DATA_TAG}] Loaded {len(records_p3)} records from {PKL_PATH}')

bc_map = defaultdict(list)
if BC_PATH and os.path.exists(BC_PATH):
    bc_raw = json.load(open(BC_PATH))
    for out_key, sel in bc_raw.items():
        step_idx = sel.get('step_idx', 0)
        suffix = f'_step{step_idx}'
        qid = out_key[:-len(suffix)] if out_key.endswith(suffix) else out_key
        bc_map[qid].append({'step_idx': step_idx, 'ff_soft': sel.get('ff_soft'),
                             'ff_hard': sel.get('ff_hard', False), 'condition': sel.get('condition')})
    print(f'Loaded best_conditions for {len(bc_map)} question(s) from {BC_PATH}')
else:
    for rec in records_p3:
        qid = rec['question_id']
        bc_map[qid].append({'step_idx': rec.get('step_idx', 0), 'ff_soft': rec.get('ff_soft'),
                             'ff_hard': rec.get('ff_hard', True), 'condition': rec.get('condition')})
    print('Built bc_map from pkl records (no best_conditions JSON available)')

rec_by_qid = defaultdict(list)
for rec in records_p3:
    rec_by_qid[rec['question_id']].append(rec)

print(f'Ready. DATA_TAG={DATA_TAG!r}')

# ── Cell 11: within-question rank test ───────────────────────────────────────
print('\n── PART 3 — STRONGER TEST 2: within-question rank test ──────────────────────')
rank_rows, ranks_of_salient = [], []

for qid in sorted(bc_map):
    valid_steps = [s for s in bc_map[qid] if s['ff_soft'] is not None]
    if not valid_steps:
        continue
    salient = max(valid_steps, key=lambda s: s['ff_soft'])
    salient_step_idx = salient['step_idx']
    salient_block    = STEP_TO_BLOCK.get(salient_step_idx, 'Final')
    salient_ff       = salient['ff_soft']

    recs_for_q = rec_by_qid.get(qid, [])
    if not recs_for_q:
        continue
    salient_rec = next(
        (r for r in recs_for_q if r.get('step_idx', 0) == salient_step_idx),
        recs_for_q[0]
    )

    deltas = {l: reasoning_delta(salient_rec, l) for l in BLOCK_LETTERS}
    if any(v is None for v in deltas.values()):
        print(f'  [SKIP] {qid} — missing delta(s)')
        continue

    sorted_blocks = sorted(deltas, key=lambda l: deltas[l], reverse=True)
    rank_of_salient = sorted_blocks.index(salient_block) + 1 if salient_block in sorted_blocks else None

    rank_rows.append({
        'question_id':     qid,
        'salient_step':    salient_step_idx,
        'salient_block':   salient_block,
        'salient_ff':      round(salient_ff, 4),
        'delta_A':         round(deltas['A'], 4),
        'delta_B':         round(deltas['B'], 4),
        'delta_C':         round(deltas['C'], 4),
        'delta_D':         round(deltas['D'], 4),
        'rank_of_salient': rank_of_salient,
    })
    if rank_of_salient is not None:
        ranks_of_salient.append(rank_of_salient)

df_rank = pd.DataFrame(rank_rows).set_index('question_id')
print(df_rank.to_string())

n_q = len(rank_rows)
if n_q > 0:
    hit_rate  = sum(1 for r in rank_rows if r['rank_of_salient'] == 1) / n_q
    mean_rank = float(np.mean(ranks_of_salient))
    print(f'\n  n={n_q}  |  Hit rate (rank=1): {hit_rate:.1%}  |  Mean rank: {mean_rank:.2f}  (null=2.5)')
    if n_q >= 6:
        diffs = [r - 2.5 for r in ranks_of_salient]
        if len(set(diffs)) > 1:
            _, p_w = stats.wilcoxon(diffs)
            print(f'  Wilcoxon signed-rank test vs null median 2.5: p={p_w:.3f}')
        else:
            print(f'  Wilcoxon skipped (all ranks identical: {ranks_of_salient})')
    else:
        print(f'  n={n_q} < 6 — Wilcoxon skipped (direction only)')

# ── Cell 12: across-question Spearman ────────────────────────────────────────
print('\n── PART 3 — STRONGER TEST 2: across-question Spearman ───────────────────────')
ff_list_p3, sal_deltas = [], []
non_sal_deltas = {l: [] for l in BLOCK_LETTERS}

for row in rank_rows:
    salient_block = row['salient_block']
    if salient_block == 'Final':
        continue
    sal_d = row.get(f'delta_{salient_block}')
    if sal_d is None or sal_d != sal_d:
        continue
    ff_list_p3.append(row['salient_ff'])
    sal_deltas.append(sal_d)
    for letter in BLOCK_LETTERS:
        if letter != salient_block:
            d = row.get(f'delta_{letter}')
            if d is not None and d == d:
                non_sal_deltas[letter].append((row['salient_ff'], d))

n_p3 = len(ff_list_p3)
if n_p3 >= 3:
    r_sal, p_sal = stats.spearmanr(ff_list_p3, sal_deltas)
    print(f'  ff_soft vs salient_delta (TARGET): r={r_sal:.3f}, p={p_sal:.3f} (n={n_p3})')
    all_ctrl_r = []
    for letter in BLOCK_LETTERS:
        pairs = non_sal_deltas[letter]
        if len(pairs) >= 3:
            ffs, ds = zip(*pairs)
            r_c, p_c = stats.spearmanr(ffs, ds)
            all_ctrl_r.append(abs(r_c))
            stronger = '  <- STRONGER than target' if abs(r_c) > abs(r_sal) else ''
            print(f'  ff_soft vs delta_{letter} (ctrl):       r={r_c:.3f}, p={p_c:.3f} (n={len(pairs)}){stronger}')
    if all_ctrl_r:
        if abs(r_sal) > max(all_ctrl_r):
            print('  -> Salient block shows strongest |r| — consistent with hypothesis.')
        else:
            print(f'  -> A non-salient block shows |r| >= salient ({abs(r_sal):.3f} vs max ctrl {max(all_ctrl_r):.3f}).')
    print(f'  Null delta band: [{rand_deltas.mean()-rand_deltas.std():.2f}, {rand_deltas.mean()+rand_deltas.std():.2f}]')
else:
    print(f'  n={n_p3} — not enough data for Spearman.')

# ── Cell 13: Full Test 3 — conclusion probe ───────────────────────────────────
print('\n── PART 3 — FULL TEST 3: Conclusion-probe cosine coupling ───────────────────')
conc_cov = {l: 0 for l in BLOCK_LETTERS}
for rec in records_p3:
    ce = rec.get('conclusion_embeddings', {})
    for l in BLOCK_LETTERS:
        if ce.get(l) is not None:
            conc_cov[l] += 1

print(f'Conclusion embedding coverage across {len(records_p3)} records:')
for l in BLOCK_LETTERS:
    print(f'  Block {l}: {conc_cov[l]}/{len(records_p3)}  ({100*conc_cov[l]/len(records_p3):.0f}%)')

usable_recs = [rec for rec in records_p3
               if any(rec.get('conclusion_embeddings', {}).get(l) is not None
                      for l in BLOCK_LETTERS)]
if not usable_recs:
    print('No conclusion embeddings — Test 3 deferred.')
else:
    print(f'{len(usable_recs)} record(s) have at least one Conclusion embedding.')
    cos_rows = []
    for rec in sorted(usable_recs, key=lambda x: x.get('ff_soft', 0)):
        ae = rec['anchor_embeddings']
        ce = rec.get('conclusion_embeddings', {})
        row = {'question_id': rec['question_id'], 'step_idx': rec.get('step_idx', 0),
               'ff_soft': round(rec.get('ff_soft', 0), 4)}
        for letter in BLOCK_LETTERS:
            reas = ae.get(f'Reasoning_{letter}')
            conc = ce.get(letter)
            row[f'cos_{letter}'] = round(cosine_sim(reas, conc), 4) if reas is not None and conc is not None else None
        cos_rows.append(row)

    print(pd.DataFrame(cos_rows).set_index('question_id').to_string())

    for letter in BLOCK_LETTERS:
        paired = [(r['ff_soft'], r[f'cos_{letter}'])
                  for r in cos_rows if r.get(f'cos_{letter}') is not None]
        if len(paired) >= 3:
            ffs, css = zip(*paired)
            r_c, p_c = stats.spearmanr(ffs, css)
            print(f'  Spearman ff_soft vs cos_{letter}: r={r_c:.3f}, p={p_c:.3f} (n={len(paired)})')
        else:
            print(f'  Spearman ff_soft vs cos_{letter}: n={len(paired)} — skipped')

    print(f'\nNull cosine band: [{rand_cosines.mean()-rand_cosines.std():.4f}, {rand_cosines.mean()+rand_cosines.std():.4f}]')
    all_cos = [r[f'cos_{l}'] for r in cos_rows for l in BLOCK_LETTERS if r.get(f'cos_{l}') is not None]
    if all_cos:
        within = sum(1 for c in all_cos
                     if rand_cosines.mean()-rand_cosines.std() < c < rand_cosines.mean()+rand_cosines.std())
        print(f'Observed cosines within null band: {within}/{len(all_cos)}')

# ══════════════════════════════════════════════════════════════════════════════
# PART 2b — LAYER SWEEP: Test 2 and Test 3 at each extracted layer
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 80)
print('PART 2b — LAYER SWEEP: Test 2 + Test 3 at each layer')
print('=' * 80)

# Detect which layers are present in the pkl
sweep_layers_present = sorted(set(
    int(k.split('_L')[1])
    for rec in records_p3
    for k in rec
    if k.startswith('anchor_embeddings_L')
))

if not sweep_layers_present:
    print('[INFO] No multi-layer embeddings found in pkl (old format or --sweep_layers not used).')
    print('[INFO] Re-run extract_fur_embeddings.py with default --sweep_layers 4 8 20 28 to enable.')
else:
    print(f'Layers present in pkl: {sweep_layers_present}')
    print()

    sweep_table = []  # one row per layer

    for layer_n in sweep_layers_present:
        anc_key  = f'anchor_embeddings_L{layer_n}'
        conc_key = f'conclusion_embeddings_L{layer_n}'

        # ── Rebuild rand_deltas / rand_cosines for this layer ─────────────────
        pool_l = []
        for rec in records_p3:
            ae_l = rec.get(anc_key, {})
            for k in TRAJ_KEYS:
                v = ae_l.get(k)
                if v is not None:
                    pool_l.append(v.astype(np.float32))
        if len(pool_l) < 10:
            print(f'  Layer {layer_n}: insufficient anchor vectors ({len(pool_l)}) — skipping')
            continue
        pool_l = np.stack(pool_l)
        rng_l = np.random.default_rng(42)
        rd_l, rc_l = [], []
        for _ in range(1000):
            i, j = rng_l.choice(len(pool_l), size=2, replace=False)
            a, b = pool_l[i], pool_l[j]
            rc_l.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
            rd_l.append(float(np.linalg.norm(a - b)))
        rand_deltas_l  = np.array(rd_l)
        rand_cosines_l = np.array(rc_l)

        # ── Test 2: within-question rank test for this layer ──────────────────
        rank_rows_l, ranks_sal_l = [], []
        for qid in sorted(bc_map):
            valid_steps = [s for s in bc_map[qid] if s['ff_soft'] is not None]
            if not valid_steps:
                continue
            salient = max(valid_steps, key=lambda s: s['ff_soft'])
            salient_step_idx = salient['step_idx']
            salient_block    = STEP_TO_BLOCK.get(salient_step_idx, 'Final')
            salient_ff       = salient['ff_soft']

            recs_for_q = rec_by_qid.get(qid, [])
            if not recs_for_q:
                continue
            salient_rec = next(
                (r for r in recs_for_q if r.get('step_idx', 0) == salient_step_idx),
                recs_for_q[0]
            )

            ae_l = salient_rec.get(anc_key, {})
            deltas_l = {}
            for letter in BLOCK_LETTERS:
                a = ae_l.get(f'Answer_{letter}')
                r = ae_l.get(f'Reasoning_{letter}')
                if a is not None and r is not None:
                    deltas_l[letter] = float(np.linalg.norm(
                        r.astype(np.float32) - a.astype(np.float32)
                    ))
            if len(deltas_l) < 4:
                continue
            sorted_b = sorted(deltas_l, key=lambda l: deltas_l[l], reverse=True)
            rank_s = sorted_b.index(salient_block) + 1 if salient_block in sorted_b else None
            if rank_s is not None:
                rank_rows_l.append({'salient_ff': salient_ff, 'salient_block': salient_block,
                                    'rank': rank_s,
                                    **{f'd_{l}': deltas_l[l] for l in BLOCK_LETTERS}})
                ranks_sal_l.append(rank_s)

        n_ql = len(rank_rows_l)
        hit_rate_l  = sum(1 for r in rank_rows_l if r['rank'] == 1) / n_ql if n_ql > 0 else float('nan')
        mean_rank_l = float(np.mean(ranks_sal_l)) if ranks_sal_l else float('nan')

        # Spearman r (ff_soft vs salient delta)
        r_sal_l = float('nan')
        ff_sp, sd_sp = [], []
        for row in rank_rows_l:
            sb = row['salient_block']
            if sb == 'Final':
                continue
            d = row.get(f'd_{sb}')
            if d is not None and d == d:
                ff_sp.append(row['salient_ff'])
                sd_sp.append(d)
        if len(ff_sp) >= 3:
            r_sal_l, _ = stats.spearmanr(ff_sp, sd_sp)

        # ── Test 3: mean cosine (conclusion probe) for this layer ─────────────
        cos_vals_l = []
        for rec in records_p3:
            ce_l = rec.get(conc_key, {})
            ae_l = rec.get(anc_key, {})
            for letter in BLOCK_LETTERS:
                reas = ae_l.get(f'Reasoning_{letter}')
                conc = ce_l.get(letter)
                if reas is not None and conc is not None:
                    cos_vals_l.append(cosine_sim(reas, conc))
        mean_cos_l = float(np.mean(cos_vals_l)) if cos_vals_l else float('nan')
        null_lo_l = rand_cosines_l.mean() - rand_cosines_l.std()
        null_hi_l = rand_cosines_l.mean() + rand_cosines_l.std()
        within_null_l = (
            f'{sum(1 for c in cos_vals_l if null_lo_l < c < null_hi_l)}/{len(cos_vals_l)}'
            if cos_vals_l else 'n/a'
        )

        primary_flag = ' *' if layer_n == 14 else ''
        sweep_table.append({
            'Layer':           f'L{layer_n}{primary_flag}',
            'n_q':             n_ql,
            'T2 hit rate':     f'{hit_rate_l:.1%}' if not np.isnan(hit_rate_l) else 'n/a',
            'T2 mean rank':    f'{mean_rank_l:.2f}' if not np.isnan(mean_rank_l) else 'n/a',
            'T2 Spearman r':   f'{r_sal_l:.3f}' if not np.isnan(r_sal_l) else 'n/a',
            'T3 mean cosine':  f'{mean_cos_l:.4f}' if not np.isnan(mean_cos_l) else 'n/a',
            'T3 within null':  within_null_l,
            'null_delta_band': f'[{rand_deltas_l.mean()-rand_deltas_l.std():.2f},{rand_deltas_l.mean()+rand_deltas_l.std():.2f}]',
        })

    if sweep_table:
        df_sweep = pd.DataFrame(sweep_table).set_index('Layer')
        print(df_sweep.to_string())
        print()
        print('* = primary layer (14) — backward-compat anchor_embeddings key')
        print()

        # Identify strongest layer by Test 2 hit rate
        numeric_rows = [(r['Layer'], float(r['T2 hit rate'].rstrip('%'))/100,
                         float(r['T2 Spearman r']) if r['T2 Spearman r'] != 'n/a' else 0.0)
                        for r in sweep_table
                        if r['T2 hit rate'] != 'n/a' and r['T2 Spearman r'] != 'n/a']
        if numeric_rows:
            best_hr  = max(numeric_rows, key=lambda x: x[1])
            best_spr = max(numeric_rows, key=lambda x: abs(x[2]))
            print(f'Strongest hit rate:    {best_hr[0]}  ({best_hr[1]:.1%})')
            print(f'Strongest |Spearman|:  {best_spr[0]}  (r={best_spr[2]:.3f})')
            if best_hr[0] == best_spr[0]:
                print(f'  → Both metrics agree: {best_hr[0]} is most informative for reasoning geometry.')
            else:
                print(f'  → Hit rate and Spearman disagree — review full table above.')

print('\n' + '=' * 80)
print('ANALYSIS COMPLETE')
print(f'Plot saved to: {args.outdir}/')
print('=' * 80)
