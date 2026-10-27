#!/usr/bin/env python
"""InsightForge AI Step 22: paired recent-history RF experiment.

Reads electricity_data.zip directly. Imports, but never edits, step20_strict_holdout.py.
Writes new artifacts only under step22_outputs/.
"""
from __future__ import annotations
import csv, json, math, time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import step20_strict_holdout as baseline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'step22_outputs'
WINDOWS = baseline.WINDOWS[1:]
SETTINGS = dict(baseline.MODEL_SETTINGS)
REPS = 10_000
BLOCKS = (12, 24, 48)
PRIMARY_BLOCK = 24
MIN_EVENT_N = 10
MIN_VALID_EVENT_REPS = 9_500
BOOT_SEED = 20261022
BASE_NAMES = ['lag24','lag168','hour_sin','hour_cos','weekday_sin','weekday_cos','is_weekend','month']
NEW_NAMES = ['lag1','lag2','lag3','lag1_minus_lag2','lag1_minus_lag3','mean_previous_3h','mean_previous_6h']


def candidate_features(t, hourly):
    base = baseline.features(t, hourly)
    if base is None:
        return None
    ts = [t - timedelta(hours=i) for i in range(1, 7)]
    if any(x not in hourly for x in ts):
        return None
    v = [hourly[x] for x in ts]
    a,b,c,d,e,f = v
    return base + [a,b,c,a-b,a-c,(a+b+c)/3.0,(a+b+c+d+e+f)/6.0]


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return {'n': 0, 'mae': None, 'rmse': None, 'bias': None}
    err = p-y
    return {'n': int(len(y)), 'mae': float(mean_absolute_error(y,p)),
            'rmse': float(np.sqrt(mean_squared_error(y,p))), 'bias': float(np.mean(err))}


def compare(y, ref, cand, event):
    r, c = metrics(y,ref), metrics(y,cand)
    def rel(num, den): return None if den == 0 else float(num/den*100)
    overall = {'reference':r,'candidate':c,
        'mae_improvement_pct':rel(r['mae']-c['mae'],r['mae']),
        'rmse_worsening_pct':rel(c['rmse']-r['rmse'],r['rmse']),
        'absolute_bias_difference_kw':abs(c['bias'])-abs(r['bias']),
        'bias_difference_kw':c['bias']-r['bias']}
    m = np.asarray(event,bool)
    if not m.any():
        ev={'n_events':0,'reference':metrics([],[]),'candidate':metrics([],[]),
            'mae_improvement_pct':None,'rmse_worsening_pct':None,
            'absolute_bias_difference_kw':None,'bias_difference_kw':None}
    else:
        re,ce=metrics(np.asarray(y)[m],np.asarray(ref)[m]),metrics(np.asarray(y)[m],np.asarray(cand)[m])
        ev={'n_events':int(m.sum()),'reference':re,'candidate':ce,
            'mae_improvement_pct':rel(re['mae']-ce['mae'],re['mae']),
            'rmse_worsening_pct':rel(ce['rmse']-re['rmse'],re['rmse']),
            'absolute_bias_difference_kw':abs(ce['bias'])-abs(re['bias']),
            'bias_difference_kw':ce['bias']-re['bias']}
    return {'overall':overall,'upward_event':ev}


def event_threshold(hourly, start):
    changes=[]
    for t in sorted(hourly):
        prev=t-timedelta(hours=1)
        if t < start and prev in hourly:
            changes.append(abs(hourly[t]-hourly[prev]))
    if len(changes)<2: raise RuntimeError(f'Insufficient pre-window changes for {start}')
    return float(np.quantile(changes, .90)), len(changes)


def contiguous_runs(records):
    """Return GLOBAL indices for maximal consecutive-hour runs, within each window."""
    groups=defaultdict(list)
    for i,r in enumerate(records): groups[r['window']].append((i,r))
    runs=[]
    for w in sorted(groups):
        items=groups[w]; start=0
        for j in range(1,len(items)):
            if items[j][1]['timestamp']-items[j-1][1]['timestamp'] != timedelta(hours=1):
                runs.append(np.array([x[0] for x in items[start:j]],dtype=int)); start=j
        if items: runs.append(np.array([x[0] for x in items[start:]],dtype=int))
    return runs


def draw_indices(runs, block_len, rng):
    pieces=[]; short_runs=0; short_obs=0
    for run in runs:
        n=len(run); L=min(block_len,n)
        if n<block_len: short_runs+=1; short_obs+=n
        sample=[]
        while len(sample)<n:
            s=int(rng.integers(0,n-L+1))
            sample.extend(run[s:s+L].tolist()[:n-len(sample)])
        pieces.extend(sample)
    return np.asarray(pieces,dtype=int),short_runs,short_obs


def bootstrap_stats(y, r, c, ev):
    er=np.asarray(r)-np.asarray(y); ec=np.asarray(c)-np.asarray(y)
    mr=float(np.mean(np.abs(er))); mc=float(np.mean(np.abs(ec)))
    rr=float(np.sqrt(np.mean(er**2))); rc=float(np.sqrt(np.mean(ec**2)))
    br=float(np.mean(er)); bc=float(np.mean(ec))
    out={'overall_mae_improvement_pct':(mr-mc)/mr*100 if mr else np.nan,
         'overall_rmse_worsening_pct':(rc-rr)/rr*100 if rr else np.nan,
         'overall_absolute_bias_difference_kw':abs(bc)-abs(br),
         'overall_bias_difference_kw':bc-br}
    m=np.asarray(ev,bool); n=int(m.sum()); out['event_n']=n
    if n>=MIN_EVENT_N:
        re,ce=er[m],ec[m]; emr=float(np.mean(np.abs(re))); emc=float(np.mean(np.abs(ce)))
        err=float(np.sqrt(np.mean(re**2))); ecr=float(np.sqrt(np.mean(ce**2)))
        ebr=float(np.mean(re)); ebc=float(np.mean(ce))
        out.update({'event_mae_improvement_pct':(emr-emc)/emr*100 if emr else np.nan,
          'event_rmse_worsening_pct':(ecr-err)/err*100 if err else np.nan,
          'event_absolute_bias_difference_kw':abs(ebc)-abs(ebr),
          'event_bias_difference_kw':ebc-ebr})
    else:
        out.update({k:np.nan for k in ['event_mae_improvement_pct','event_rmse_worsening_pct','event_absolute_bias_difference_kw','event_bias_difference_kw']})
    return out


def ci(vals):
    a=np.asarray(vals,float); a=a[np.isfinite(a)]
    if not len(a): return {'low':None,'high':None,'n_finite':0}
    lo,hi=np.percentile(a,[2.5,97.5])
    return {'low':float(lo),'high':float(hi),'n_finite':int(len(a))}


def run_bootstrap(records, block_len, seed):
    rng=np.random.default_rng(seed); runs=contiguous_runs(records)
    y=np.array([x['actual'] for x in records],float)
    r=np.array([x['reference_prediction'] for x in records],float)
    c=np.array([x['candidate_prediction'] for x in records],float)
    ev=np.array([x['is_upward_event'] for x in records],bool)
    keys=['overall_mae_improvement_pct','overall_rmse_worsening_pct','overall_absolute_bias_difference_kw','overall_bias_difference_kw','event_mae_improvement_pct','event_rmse_worsening_pct','event_absolute_bias_difference_kw','event_bias_difference_kw']
    samples={k:[] for k in keys}; valid=0; short_run_count=0; short_obs=0
    for b in range(REPS):
        idx,sr,so=draw_indices(runs,block_len,rng)
        short_run_count=max(short_run_count,sr); short_obs=max(short_obs,so)
        z=bootstrap_stats(y[idx],r[idx],c[idx],ev[idx])
        if z['event_n']>=MIN_EVENT_N: valid+=1
        for k in keys:
            if np.isfinite(z[k]): samples[k].append(float(z[k]))
        if (b+1)%1000==0: print(f'  Bootstrap {block_len}h: {b+1:,}/{REPS:,}')
    return {'requested_block_length_hours':block_len,'replicates_planned':REPS,
      'valid_event_replicates':valid,'insufficient_event_replicates':REPS-valid,
      'valid_event_replicate_fraction':valid/REPS,'event_bootstrap_reliable':valid>=MIN_VALID_EVENT_REPS,
      'contiguous_run_count':len(runs),'contiguous_run_min_hours':min(map(len,runs)) if runs else None,
      'contiguous_run_max_hours':max(map(len,runs)) if runs else None,
      'short_run_count_per_replicate':short_run_count,'short_run_observations_per_replicate':short_obs,
      'intervals':{k:ci(v) for k,v in samples.items()}}


def decide(point, interval, threshold, direction, reliable=True):
    if not reliable or point is None or interval is None or interval['low'] is None: return 'INCONCLUSIVE'
    lo,hi=interval['low'],interval['high']
    if direction=='min':
        if point<threshold: return 'FAIL'
        if lo>=threshold: return 'PASS'
        if hi<threshold: return 'FAIL'
        return 'INCONCLUSIVE'
    if point>threshold: return 'FAIL'
    if hi<=threshold: return 'PASS'
    if lo>threshold: return 'FAIL'
    return 'INCONCLUSIVE'


def make_decisions(m, bs):
    o=m['overall']; e=m['upward_event']; ci=bs['intervals']; reliable=bs['event_bootstrap_reliable']
    specs=[('1_overall_mae_improvement','Overall MAE improvement (%)',o['mae_improvement_pct'],'overall_mae_improvement_pct',3,'min',False),
      ('2_overall_rmse_worsening','Overall RMSE worsening (%)',o['rmse_worsening_pct'],'overall_rmse_worsening_pct',1,'max',False),
      ('3_overall_absolute_bias_difference','Overall absolute-bias difference (kW)',o['absolute_bias_difference_kw'],'overall_absolute_bias_difference_kw',.05,'max',False),
      ('4_event_mae_improvement','Upward-event MAE improvement (%)',e['mae_improvement_pct'],'event_mae_improvement_pct',5,'min',True),
      ('5_event_absolute_bias_difference','Upward-event absolute-bias difference (kW)',e['absolute_bias_difference_kw'],'event_absolute_bias_difference_kw',.05,'max',True),
      ('6_event_bias_direction','Upward-event bias difference (kW)',e['bias_difference_kw'],'event_bias_difference_kw',-.05,'min',True)]
    out=[]
    for key,label,point,ci_key,threshold,direction,is_event in specs:
        out.append({'criterion':key,'label':label,'point_estimate':point,'threshold':threshold,'direction':direction,
          'ci_95':ci[ci_key],'decision':decide(point,ci[ci_key],threshold,direction,reliable if is_event else True),
          'event_bootstrap_reliable':reliable if is_event else None})
    return out


def safe(o):
    if isinstance(o,dict): return {str(k):safe(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [safe(x) for x in o]
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o,float) and not math.isfinite(o): return None
    return o


def write_report(s,path):
    lines=['# InsightForge AI — Step 22 Recent-History Experiment','',
      '## Interpretation','',
      'Retrospective exploratory comparison on Windows 2–4; this is not independent confirmation. The primary decision uses pooled paired observations, not the unweighted average of window metrics.','',
      '## Protocol','',
      '- Reference: frozen eight-feature Random Forest. Candidate: same eight features plus seven recent-history features.','- Same shared training cohort and paired target timestamps in each window; training timestamps strictly precede each window.','- Exact prior-hour timestamps required; no imputation, gap skipping, or shortened rolling windows.','- Mean bias = forecast minus actual.','- Paired moving-block bootstrap: 10,000 replicates, 95% percentile intervals; 24h primary, 12h/48h sensitivity.','- Upward events use positive hourly increase greater than the pre-window 90th percentile of absolute consecutive-hour changes; labels remain fixed during resampling.','',
      '## Window-level counts','', '| Window | Ref eligible eval | Paired eval | Ref-eligible train | Shared train | Event n |','|---|---:|---:|---:|---:|---:|']
    for w in s['window_results']:
        lines.append(f"| {w['window']} | {w['evaluation_reference_eligible']} | {w['evaluation_paired']} | {w['training_reference_eligible']} | {w['training_shared']} | {w['event_n']} |")
    lines += ['', '## Per-window metrics','', '| Window | Model | n | MAE kW | RMSE kW | Bias kW |','|---|---|---:|---:|---:|---:|']
    for w in s['window_results']:
        for label,k in [('Reference','reference'),('Candidate','candidate')]:
            x=w['metrics']['overall'][k]; lines.append(f"| {w['window']} | {label} | {x['n']} | {x['mae']:.6f} | {x['rmse']:.6f} | {x['bias']:.6f} |")
    o=s['pooled_metrics']['overall']; e=s['pooled_metrics']['upward_event']
    lines += ['', '## Pooled paired primary results (Windows 2–4)','', '| Metric | Reference | Candidate | Comparison |','|---|---:|---:|---|',
      f"| Overall MAE (kW) | {o['reference']['mae']:.6f} | {o['candidate']['mae']:.6f} | improvement {o['mae_improvement_pct']:.3f}% |",
      f"| Overall RMSE (kW) | {o['reference']['rmse']:.6f} | {o['candidate']['rmse']:.6f} | worsening {o['rmse_worsening_pct']:.3f}% |",
      f"| Overall bias (kW) | {o['reference']['bias']:.6f} | {o['candidate']['bias']:.6f} | difference {o['bias_difference_kw']:.6f} kW |",
      f"| Overall absolute-bias difference | — | — | {o['absolute_bias_difference_kw']:.6f} kW |",'',
      f"Upward-event observations: **{e['n_events']}**."]
    if e['n_events']:
        lines += [f"- Event MAE: reference {e['reference']['mae']:.6f} kW; candidate {e['candidate']['mae']:.6f} kW; relative improvement {e['mae_improvement_pct']:.3f}%.",
          f"- Event bias: reference {e['reference']['bias']:.6f} kW; candidate {e['candidate']['bias']:.6f} kW.",
          f"- Event absolute-bias difference {e['absolute_bias_difference_kw']:.6f} kW; event bias difference {e['bias_difference_kw']:.6f} kW."]
    lines += ['', '## Six locked acceptance decisions','', '| Criterion | Point | Threshold | 95% CI | Decision |','|---|---:|---:|---|---|']
    for d in s['decisions']:
        p='n/a' if d['point_estimate'] is None else f"{d['point_estimate']:.6f}"
        ci=d['ci_95']; ci_s='unavailable' if ci['low'] is None else f"[{ci['low']:.6f}, {ci['high']:.6f}]"
        lines.append(f"| {d['label']} | {p} | {d['direction']} {d['threshold']} | {ci_s} | **{d['decision']}** |")
    statuses=[d['decision'] for d in s['decisions']]
    outcome='PASS — all six criteria passed' if all(x=='PASS' for x in statuses) else ('FAIL — at least one criterion failed' if 'FAIL' in statuses else 'INCONCLUSIVE — at least one criterion remains inconclusive')
    lines += ['',f'**Overall outcome: {outcome}.**','', '## Bootstrap sensitivity','', '| Block | Valid event reps | Insufficient event reps | Event reliable | Short runs/replicate |','|---:|---:|---:|---|---:|']
    for b in ('12','24','48'):
        x=s['bootstrap'][b]; lines.append(f"| {b} h | {x['valid_event_replicates']} | {x['insufficient_event_replicates']} | {x['event_bootstrap_reliable']} | {x['short_run_count_per_replicate']} |")
    lines += ['', 'Short runs use the full run as the maximum effective block; no blocks cross timestamp gaps or evaluation-window boundaries. Event-level decisions are INCONCLUSIVE if fewer than 9,500 of 10,000 replicates contain at least 10 events. Intervals are conditional on observed periods; separate intervals do not give a family-wise 95% guarantee.','', '## Limitations','', '- Windows 2–4 are retrospective exploratory periods, not independent confirmation.','- Feature importance is descriptive, not causal.','- Do not claim overall success if any criterion fails or remains inconclusive.','', 'Full paired predictions: `step22_paired_predictions.csv`; machine-readable results: `step22_summary.json`.','']
    path.write_text('\n'.join(lines),encoding='utf-8')


def main():
    t0=time.time(); OUT.mkdir(exist_ok=True)
    print('InsightForge AI Step 22 — controlled paired recent-history experiment')
    print(f'Frozen RF settings: {SETTINGS}')
    print(f'Outputs: {OUT}')
    hourly=baseline.load_hourly_from_zip(ROOT/'electricity_data.zip')
    records=[]; windows=[]
    for name,start,end in WINDOWS:
        print(f'\n=== {name}: {start} through {end} ===')
        threshold, n_changes=event_threshold(hourly,start)
        tr_ref_n=0; trx=[]; tcy=[]; ty=[]
        for t in sorted(x for x in hourly if x<start):
            bx=baseline.features(t,hourly)
            if bx is not None: tr_ref_n+=1
            cx=candidate_features(t,hourly)
            if cx is not None:
                trx.append(cx[:len(BASE_NAMES)]); tcy.append(cx); ty.append(hourly[t])
        eval_ref_n=0; times=[]; exr=[]; exc=[]; y=[]
        for t in sorted(x for x in hourly if start<=x<=end):
            bx=baseline.features(t,hourly)
            if bx is not None: eval_ref_n+=1
            cx=candidate_features(t,hourly)
            if cx is not None:
                times.append(t); exr.append(cx[:len(BASE_NAMES)]); exc.append(cx); y.append(hourly[t])
        if not ty or not y: raise RuntimeError(f'No shared train/evaluation cohort in {name}')
        print(f'Train baseline-eligible={tr_ref_n:,}; shared={len(ty):,}; excluded={tr_ref_n-len(ty):,}')
        print(f'Eval baseline-eligible={eval_ref_n:,}; paired={len(y):,}; excluded={eval_ref_n-len(y):,}; event threshold={threshold:.6f} kW')
        yr=np.asarray(ty,np.float32)
        rm=RandomForestRegressor(**SETTINGS); cm=RandomForestRegressor(**SETTINGS)
        rm.fit(np.asarray(trx,np.float32),yr); cm.fit(np.asarray(tcy,np.float32),yr)
        rp=rm.predict(np.asarray(exr,np.float32)).astype(float); cp=cm.predict(np.asarray(exc,np.float32)).astype(float)
        ev=[]
        for t in times:
            # candidate feature eligibility ensures t-1 exists exactly.
            ev.append(hourly[t]-hourly[t-timedelta(hours=1)] > threshold)
        wm=compare(y,rp,cp,ev)
        print(f"Reference MAE/RMSE={wm['overall']['reference']['mae']:.4f}/{wm['overall']['reference']['rmse']:.4f}; candidate={wm['overall']['candidate']['mae']:.4f}/{wm['overall']['candidate']['rmse']:.4f}; events={sum(ev)}")
        for i,t in enumerate(times):
            records.append({'window':name,'timestamp':t,'actual':float(y[i]),'reference_prediction':float(rp[i]),'candidate_prediction':float(cp[i]),'is_upward_event':bool(ev[i]),'event_threshold_kw':threshold})
        windows.append({'window':name,'start':start.isoformat(sep=' '),'end':end.isoformat(sep=' '),'event_threshold_kw':threshold,'historical_consecutive_changes_for_threshold':n_changes,
          'training_reference_eligible':tr_ref_n,'training_shared':len(ty),'training_excluded_from_shared':tr_ref_n-len(ty),
          'evaluation_reference_eligible':eval_ref_n,'evaluation_paired':len(y),'evaluation_excluded_from_paired':eval_ref_n-len(y),'event_n':int(sum(ev)),'metrics':wm,
          'reference_feature_importance':sorted([{'feature':n,'importance':float(v)} for n,v in zip(BASE_NAMES,rm.feature_importances_)],key=lambda z:z['importance'],reverse=True),
          'candidate_feature_importance':sorted([{'feature':n,'importance':float(v)} for n,v in zip(BASE_NAMES+NEW_NAMES,cm.feature_importances_)],key=lambda z:z['importance'],reverse=True)})
        del rm,cm
    y=np.array([x['actual'] for x in records]); r=np.array([x['reference_prediction'] for x in records]); c=np.array([x['candidate_prediction'] for x in records]); ev=np.array([x['is_upward_event'] for x in records])
    pooled=compare(y,r,c,ev)
    print(f"\nPooled paired n={len(records):,}; reference MAE={pooled['overall']['reference']['mae']:.5f}; candidate MAE={pooled['overall']['candidate']['mae']:.5f}")
    bs={}
    for block in BLOCKS:
        print(f'\nBootstrap block {block}h...')
        bs[str(block)]=run_bootstrap(records,block,BOOT_SEED+block)
    decisions=make_decisions(pooled,bs[str(PRIMARY_BLOCK)])
    for d in decisions: print(f"{d['criterion']}: {d['decision']} (estimate={d['point_estimate']}, threshold={d['threshold']})")
    statuses=[d['decision'] for d in decisions]
    outcome='PASS' if all(s=='PASS' for s in statuses) else ('FAIL' if 'FAIL' in statuses else 'INCONCLUSIVE')
    summary={'experiment':'InsightForge AI Step 22 recent-history experiment','status':'completed','interpretation':'retrospective exploratory comparison; not independent confirmation',
      'reference_script':'step20_strict_holdout.py (read-only dependency)','source_zip':'electricity_data.zip (read directly; not extracted)',
      'evaluation_windows':[{'name':n,'start':a.isoformat(sep=' '),'end':b.isoformat(sep=' ')} for n,a,b in WINDOWS],
      'reference_feature_names':BASE_NAMES,'added_candidate_feature_names':NEW_NAMES,'model_settings':SETTINGS,
      'primary_population':'pooled paired evaluation observations from Windows 2-4','pooled_paired_n':len(records),'pooled_metrics':pooled,
      'window_results':windows,'bootstrap':bs,'primary_bootstrap_block_length_hours':PRIMARY_BLOCK,
      'decisions':decisions,'overall_outcome':outcome,'runtime_seconds':round(time.time()-t0,3),'bootstrap_seed':BOOT_SEED,
      'bootstrap_replicates_per_block_length':REPS,'event_minimum_observations_per_replicate':MIN_EVENT_N,'minimum_valid_event_replicates':MIN_VALID_EVENT_REPS,
      'notes':['Shared training cohort requires both feature sets.','All added features require exact preceding timestamps; no imputation.','Same model settings and paired evaluation timestamps per window.','Blocks do not cross timestamp gaps or evaluation-window boundaries.','Event labels are fixed during resampling.','Confidence intervals are conditional on observed evaluation periods; no family-wise guarantee is implied.']}
    # Write full paired predictions with exact timestamps.
    pred_path=OUT/'step22_paired_predictions.csv'
    with pred_path.open('w',newline='',encoding='utf-8') as f:
        fields=['window','timestamp','actual','reference_prediction','candidate_prediction','reference_error','candidate_error','is_upward_event','event_threshold_kw']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for x in records:
            w.writerow({'window':x['window'],'timestamp':x['timestamp'].isoformat(sep=' '),'actual':f"{x['actual']:.10f}",'reference_prediction':f"{x['reference_prediction']:.10f}",'candidate_prediction':f"{x['candidate_prediction']:.10f}",'reference_error':f"{x['reference_prediction']-x['actual']:.10f}",'candidate_error':f"{x['candidate_prediction']-x['actual']:.10f}",'is_upward_event':int(x['is_upward_event']),'event_threshold_kw':f"{x['event_threshold_kw']:.10f}"})
    (OUT/'step22_summary.json').write_text(json.dumps(safe(summary),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    write_report(summary,OUT/'STEP22_RECENT_HISTORY_RESULTS.md')
    print(f'\nOVERALL OUTCOME: {outcome}')
    print('New outputs: step22_outputs/STEP22_RECENT_HISTORY_RESULTS.md, step22_outputs/step22_summary.json, step22_outputs/step22_paired_predictions.csv')
    print('Frozen files and source ZIP were not modified.')
    print(f'Runtime: {time.time()-t0:.1f}s')

if __name__=='__main__':
    main()
