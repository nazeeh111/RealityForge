"""Generate presentation tables/figures from saved measurements; never invent scores."""
import json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/realityforge-matplotlib')
os.environ.setdefault('MPL_IGNORE_SYSTEM_FONTS','1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/assets';ASSETS.mkdir(parents=True,exist_ok=True)
LABELS={'hybrid':'Learned + physics','neural':'Learned initializer','analytical':'Analytical','analytical_refine':'Analytical + physics','constant_velocity':'Constant velocity','iris_recurrence':'IRIS recurrence adapter'}
COLORS={'hybrid':'#386b58','neural':'#94b3a0','analytical':'#b5b0a2','analytical_refine':'#527eaf','constant_velocity':'#d2b68d','iris_recurrence':'#d57856'}


def main():
    synth=json.loads((ROOT/'benchmarks/synthetic.json').read_text())
    real=json.loads((ROOT/'benchmarks/real.json').read_text())
    lines=['# Measured benchmark results','','Generated from the checked-in JSON reports by `python scripts/report_benchmarks.py`. Every reported failure remains in the source report. A completed/succeeded fit means a finite result was returned, not that an accuracy threshold was passed. Runtime is a single-run CPU measurement affected by host load, not a controlled speed ranking.','','## Independent synthetic trajectories','',f"{synth['protocol']['cases_per_family']} trajectories per family; seed 2026, fit 65%, forecast 35%. Clean positions come from an independent SciPy solver for pendulums and the exact sliding polynomial. Noise levels 0, 0.5 and 2 pixels; the last case per family includes 5% outliers. This layer excludes video tracking.",'','| Family | Method | Succeeded / attempted | Median forecast px | Median primary parameter error | Median inference s | Nominal interval coverage |','|---|---|---:|---:|---:|---:|---:|']
    for row in synth['summary']:
        coverage=row.get('primary_ci_coverage');error=row.get('primary_parameter_relative_error_median')
        lines.append(f"| {row['mode']} | {LABELS[row['method']]} | {row['count']-row['failures']} / {row['count']} | {row.get('forecast_rmse_px_median',float('nan')):.3f} | {'—' if error is None else f'{error*100:.3f}%'} | {row.get('runtime_s_median',float('nan')):.3f} | {'—' if coverage is None else f'{coverage*100:.1f}%'} |")
    lines+=['','The primary parameter is pendulum `k=g/L` or signed sliding acceleration. Nominal 95% intervals are uncalibrated local approximations; measured coverage is displayed above rather than assumed. Analytical initialization plus the same physical refinement is an essential ablation.','','![Synthetic forecast errors](assets/benchmark-synthetic.svg)','','## Development recording transfer check','','These five clips were excluded from neural training but informed tracker development. Errors are against extracted tracks, not independent point annotations. Families are selected explicitly; the separate selection report evaluates automatic screening.','','| Recording | Learned + physics px | Analytical + physics px | IRIS recurrence adapter px | Learned only px | Constant velocity px |','|---|---:|---:|---:|---:|---:|']
    rows=real['rows'];ids=list(dict.fromkeys(r['id'] for r in rows))
    def number(id,method):
        row=next((r for r in rows if r['id']==id and r['method']==method),None)
        return '—' if row is None else ('failed' if 'error' in row else f"{row['forecast_rmse_px']:.3f}")
    for id in ids:lines.append('| '+id+' | '+' | '.join(number(id,m) for m in ['hybrid','analytical_refine','iris_recurrence','neural','constant_velocity'])+' |')
    lines+=['','IRIS recurrence is a reimplementation of the published discrete physical component using shared observed angles. It bypasses the learned video encoder and latent objective. These numbers are not reported complete IRIS/EndPhys pipeline scores.','','| Recording | Effective length m, assuming g=9.80665 | Difference to nominal rope length | Tracking s | Inference s | Total s |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['method']!='hybrid' or 'error' in r:continue
        length=r.get('effective_length_assuming_earth_g_m');delta=r.get('relative_difference_to_nominal_rope_length')
        lines.append(f"| {r['id']} | {'—' if length is None else f'{length:.4f}'} | {'—' if delta is None else f'{delta*100:.2f}%'} | {r['tracking_runtime_s']:.3f} | {r['inference_runtime_s']:.3f} | {r['total_runtime_s']:.3f} |")
    lines+=['','Length differences are consistency checks, not precision parameter errors. The nominal rope may exclude bob radius. Independent household length is approximate and author-reported; sliding acceleration/friction truth is unavailable.','','## Frozen unseen IRIS recordings','']
    blind_path=ROOT/'benchmarks/blind-iris.json'
    if blind_path.exists():
        blind=json.loads(blind_path.read_text());lines+=['Six recording IDs and clip windows were preselected, and core/model hashes frozen before acquisition. A fixed first-frame green-component rule initializes the region. No future-frame tuning or selection is allowed. All attempts, including failures, are retained. This is an unseen-recording transfer check within the supported IRIS pendulum setup, not the complete benchmark.','','| Method | Succeeded / attempted | Median forecast px | Worst forecast px | Median inference s |','|---|---:|---:|---:|---:|']
        for method,r in blind['summary'].items():
            lines.append(f"| {LABELS.get(method,method)} | {r['succeeded']} / {r['attempted']} | {r.get('forecast_rmse_px_median',float('nan')):.3f} | {r.get('forecast_rmse_px_max',float('nan')):.3f} | {r.get('inference_runtime_s_median',float('nan')):.3f} |")
        lines+=['','See [all frozen attempts](../benchmarks/blind-iris.json) and [first-frame regions/provenance](../samples/blind-provenance.json).']
    else:lines+=['The frozen evaluation report is not present. No unseen-recording score is asserted.']
    selection=json.loads((ROOT/'benchmarks/model-selection.json').read_text())
    lines+=['','## Supported-family selection','',f"The geometry/reversal screen selected the expected family in **{selection['correct']} / {selection['count']}** listed cases. This small diagnostic includes small-angle pendulums, straight tracks and the development recordings. It does not establish arbitrary equation recognition. Short monotonic pendulum arcs remain ambiguous. [Per-case evidence](../benchmarks/model-selection.json).",'','## Read the limitations with the measurements','','See [evaluation](evaluation.md), [model card](model-card.md) and [research audit](research.md). Full latent-video baseline reproduction, calibrated uncertainty, general camera calibration and 3D reconstruction are not claimed.']
    (ROOT/'docs/benchmark-results.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'RealityForge-0.1','savefig.facecolor':'#f9f8f4','axes.facecolor':'#f9f8f4','figure.facecolor':'#f9f8f4'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,mode in zip(axes,['pendulum','sliding']):
        group=[r for r in synth['summary'] if r['mode']==mode]
        values=[max(r['forecast_rmse_px_median'],1e-4) for r in group]
        ax.barh([LABELS[r['method']] for r in group],values,color=[COLORS[r['method']] for r in group],height=.65)
        ax.set_xscale('log');ax.invert_yaxis();ax.set_xlabel('Median held-out forecast RMSE / pixels (log scale)');ax.set_title(mode.title(),loc='left',weight='bold');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.annotate(f' {v:.3f}',(v,i),va='center',fontsize=8)
        ax.set_xlim(min(values)*.5,max(values)*4)
    fig.suptitle('RealityForge / independent synthetic trajectories',fontsize=14,weight='bold',x=.02,ha='left')
    fig.savefig(ASSETS/'benchmark-synthetic.svg',metadata={'Date':None,'Creator':'RealityForge'});
    svg=ASSETS/'benchmark-synthetic.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n');fig.savefig(ASSETS/'benchmark-synthetic.png',dpi=180);plt.close(fig)
    # Update only the explicit generated block in the README.
    hybrid=[r for r in synth['summary'] if r['method']=='hybrid']
    summary=['| Synthetic family | Cases | Median forecast | Median primary parameter error |','|---|---:|---:|---:|']
    for r in hybrid:summary.append(f"| {r['mode'].title()} | {r['count']} | {r['forecast_rmse_px_median']:.3f} px | {r['primary_parameter_relative_error_median']*100:.3f}% |")
    summary+=['', 'These are trajectory-level synthetic results, with noise and outliers. They do not measure real-video accuracy.','', '| Real development recording | Held-out forecast |','|---|---:|']
    for r in rows:
        if r['method']=='hybrid':summary.append(f"| {r['id']} | {'failed' if 'error' in r else format(r['forecast_rmse_px'],'.2f')+' px'} |")
    if blind_path.exists():
        b=blind['summary']['hybrid'];summary+=['',f"A separately frozen six-recording IRIS check completed **{b['succeeded']}/{b['attempted']}** attempts, with median forecast **{b['forecast_rmse_px_median']:.2f} px**. [Every attempt and protocol](benchmarks/blind-iris.json)."]
    summary+=['','![Measured synthetic forecast comparison](docs/assets/benchmark-synthetic.svg)']
    readme=ROOT/'README.md';s=readme.read_text();start=s.index('<!-- BENCHMARKS_START -->')+len('<!-- BENCHMARKS_START -->');end=s.index('<!-- BENCHMARKS_END -->');readme.write_text(s[:start]+'\n'+'\n'.join(summary)+'\n'+s[end:])
    print('Generated benchmark-results.md, synthetic figures and README measurements.')

if __name__=='__main__':main()
