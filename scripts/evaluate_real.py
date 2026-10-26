"""Small transparent recording-level transfer check; not the official IRIS protocol."""
import argparse,json,time,platform,hashlib
from pathlib import Path
import numpy as np
from realityforge.tracking import track_video
from realityforge.inference import infer_track
ROOT=Path(__file__).resolve().parents[1]

def main(output='benchmarks/real.json'):
    manifest=json.loads((ROOT/'samples/manifest.json').read_text())['samples']
    provenance={r['id']:r for r in json.loads((ROOT/'samples/provenance.json').read_text())['records']}
    rows=[]
    for sample in manifest:
        if sample['kind']!='real':continue
        start=time.perf_counter();track=track_video(ROOT/sample['video_url'].lstrip('/'),sample['settings'].get('roi'))
        tracking_time=time.perf_counter()-start
        for method in ('hybrid','neural','analytical','analytical_refine','constant_velocity','iris_recurrence'):
            if method=='iris_recurrence' and sample['settings']['mode']!='pendulum': continue
            row={'id':sample['id'],'method':method,'tracking_runtime_s':tracking_time,'tracking_mean_confidence':float(track.confidence.mean()),'settings':sample['settings']}
            start=time.perf_counter()
            try:
                n=int(len(track.time)*.65)
                if method=='constant_velocity':
                    c=np.polyfit(track.time[:n],track.xy[:n],1);pred=track.time[:,None]*c[0]+c[1]
                    row['forecast_rmse_px']=float(np.sqrt(np.mean(np.sum((pred[n:]-track.xy[n:])**2,axis=1))))
                else:
                    result=infer_track(track,sample['settings'],method)
                    row.update(parameters=result['simulation']['parameters'],metrics=result['metrics'],warnings=result['warnings'],selected_model=result['model'])
                    row['forecast_rmse_px']=result['metrics']['forecast_rmse_px']
                    if result['model']=='pendulum':
                        length=9.80665/result['simulation']['parameters']['k']
                        nominal=provenance[sample['id']]['ground_truth'].get('rope_length_m')
                        row['effective_length_assuming_earth_g_m']=length
                        if nominal:
                            row['nominal_rope_length_m']=nominal
                            row['relative_difference_to_nominal_rope_length']=abs(length-nominal)/nominal
                    if method=='hybrid':
                        out=ROOT/'benchmarks'/'analyses';out.mkdir(exist_ok=True)
                        result['video_url']=sample['video_url'];(out/f"{sample['id']}.json").write_text(json.dumps(result,indent=2,allow_nan=False))
                row['inference_runtime_s']=time.perf_counter()-start
                row['total_runtime_s']=tracking_time+row['inference_runtime_s']
            except Exception as e:row['error']=f'{type(e).__name__}: {e}'
            rows.append(row);print(sample['id'],method,row.get('forecast_rmse_px',row.get('error')),flush=True)
    payload={'protocol':'Fit initial65% of each recording; forecast remaining35%. All physical/geometry parameters fitted only to initial segment. ROI manually selected on first frame. All recordings excluded from neural training; this small subset informed tracker development, so it is not a blind test. Errors against image tracks, not independently annotated point truth.','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'realityforge').glob('*.py')},'platform':platform.platform(),'limitations':['Small IRIS subset; not complete official protocol.','IRIS nominal rope length0.50m is not necessarily pivot-to-bob-center effective length. Relative difference is not metrology-grade parameter error.','No acceleration/friction ground truth for sliding record.','Independent household recording length only author-reported.','IRIS-inspired finite-difference baseline is the analytical method; no published GPU/video baseline reproduced.'],'rows':rows}
    path=ROOT/output;path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='benchmarks/real.json');a=p.parse_args();main(a.output)
