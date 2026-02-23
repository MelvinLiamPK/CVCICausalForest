"""Diagnostic analysis of CVCI-CF results."""
import json, os, sys, glob
import numpy as np

results_dir = sys.argv[1]

# Find all experiment directories
exp_dirs = sorted(glob.glob(os.path.join(results_dir, '*_20260218_*')))

for exp_dir in exp_dirs:
    results_path = os.path.join(exp_dir, 'results_all.json')
    meta_path = os.path.join(exp_dir, 'metadata.json')
    if not os.path.exists(results_path):
        continue
    
    with open(meta_path) as f:
        meta = json.load(f)
    with open(results_path) as f:
        results = json.load(f)
    
    name = os.path.basename(exp_dir).rsplit('_', 2)[0]
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    
    # Collect across all parameter values
    all_lambdas = []
    all_q_curves = []
    
    for param_key, sims in results.items():
        if not sims:
            continue
        for s in sims:
            cvci = s.get('cvci', {})
            lam = cvci.get('lambda_opt', float('nan'))
            qv = cvci.get('Q_values', [])
            if lam == lam:
                all_lambdas.append(lam)
            if qv:
                all_q_curves.append(qv)
    
    # --- 1. Lambda distribution ---
    if all_lambdas:
        lams = np.array(all_lambdas)
        print(f"\n  Lambda* distribution (n={len(lams)}):")
        print(f"    Mean={lams.mean():.3f}, Median={np.median(lams):.3f}, Std={lams.std():.3f}")
        print(f"    Min={lams.min():.3f}, Max={lams.max():.3f}")
        bins = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.01]
        labels = ['[0,0.05)', '[0.05,0.2)', '[0.2,0.4)', '[0.4,0.6)',
                  '[0.6,0.8)', '[0.8,0.95)', '[0.95,1]']
        counts, _ = np.histogram(lams, bins=bins)
        for label, count in zip(labels, counts):
            pct = 100 * count / len(lams)
            bar = '#' * int(pct / 2)
            print(f"    {label:>12}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # --- 2. Q(lambda) curve shape ---
    if all_q_curves:
        Q = np.array(all_q_curves)
        Q_mean = Q.mean(axis=0)
        n_lam = len(Q_mean)
        lam_grid = np.linspace(0, 1, n_lam)
        
        print(f"\n  Q(lambda) curve (averaged over {Q.shape[0]} sims):")
        for i in range(n_lam):
            denom = Q_mean.max() - Q_mean.min() + 1e-10
            bar_len = int(50 * (Q_mean[i] - Q_mean.min()) / denom)
            marker = ' <-- min' if i == np.argmin(Q_mean) else ''
            print(f"    lam={lam_grid[i]:.2f}: Q={Q_mean[i]:.6f} |{'#' * bar_len}{marker}")
        
        q_range = Q_mean.max() - Q_mean.min()
        q_scale = Q_mean.mean()
        print(f"\n    Q range: {q_range:.6f} (relative: {q_range / q_scale:.4f})")
        print(f"    Q at lam=0: {Q_mean[0]:.6f}, Q at lam=1: {Q_mean[-1]:.6f}")
        if q_range / q_scale < 0.05:
            print(f"    ** WARNING: Q curve is very flat -- lambda selection has little signal! **")
    
    # --- 3. Bias-variance decomposition for first param value ---
    first_key = list(results.keys())[0]
    sims = results[first_key]
    if sims and len(sims) >= 5:
        print(f"\n  Bias-Variance decomposition (param={first_key}, n_sims={len(sims)}):")
        for method in ['exp_only', 'obs_only', 'pooled', 'cvci']:
            preds = []
            for s in sims:
                cp = s.get(method, {}).get('cate_pred', [])
                if cp:
                    preds.append(cp)
            if len(preds) < 3:
                continue
            
            true_cate = np.array(sims[0]['true_cate'])
            P = np.array(preds)
            mean_pred = P.mean(axis=0)
            bias_sq = np.mean((mean_pred - true_cate) ** 2)
            variance = np.mean(P.var(axis=0))
            mse = np.mean(np.mean((P - true_cate[None, :]) ** 2, axis=1))
            print(f"    {method:>10}: MSE={mse:.4f} = Bias2={bias_sq:.4f} + Var={variance:.4f}")
    
    # --- 4. CATE correlation between methods (first sim) ---
    first_key = list(results.keys())[0]
    sims = results[first_key]
    if sims and len(sims) >= 1:
        s = sims[0]
        mp = {}
        for m in ['exp_only', 'obs_only', 'pooled', 'cvci']:
            cp = s.get(m, {}).get('cate_pred', [])
            if cp:
                mp[m] = np.array(cp)
        
        if len(mp) >= 2:
            print(f"\n  CATE correlation (param={first_key}, sim 0):")
            names = list(mp.keys())
            header = f"    {'':>10}" + "".join(f"{n:>10}" for n in names)
            print(header)
            for n1 in names:
                row = f"    {n1:>10}"
                for n2 in names:
                    r = np.corrcoef(mp[n1], mp[n2])[0, 1]
                    row += f"{r:>10.3f}"
                print(row)

print("\n\nDONE")
