"""Frozen recording-level IRIS test, acquired only after the estimator was fixed.

Run from any directory. --fetch downloads six specified source windows, never
whole original recordings intentionally. Existing clips are preserved. Results
include every attempted case, including failed acquisition/tracking/inference.
This is an observed-track subset study, not the official complete IRIS protocol.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REVISION='c253822f55431ca80ef2084de4bc5e79d1a488f1'
PROVENANCE=ROOT/'samples/blind-provenance.json'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_hashes():
    files=[ROOT/'realityforge'/name for name in ('physics.py','tracking.py','inference.py','estimator.py','baselines.py')]
    files += [ROOT/'models'/f'{name}.pt' for name in ('pendulum','sliding')]
    return {str(path.relative_to(ROOT)):digest(path) for path in files}


def initialize():
    if PROVENANCE.exists(): return json.loads(PROVENANCE.read_text())
    records=[]
    for angle in (20,45,90):
        for take in ('03','06'):
            source_path=f'Pendulum/pendulum_{angle}/{take}.mp4'
            records.append({'id':f'blind-pendulum{angle}-take{take}','path':f'samples/blind-iris/pendulum{angle}-take{take}.mp4','source_path':source_path,'source_url':f'https://huggingface.co/datasets/rasulkhanbayov/IRIS/resolve/{REVISION}/{source_path}','source_revision':REVISION,'source_start_seconds':2,'duration_seconds':8,'output_fps':30,'output_size':[960,540],'temporal_mapping':'source_time = clip_time + 2 seconds; playback speed unchanged','nominal_rope_length_m':.5,'license':'CC-BY-NC-4.0','license_url':'https://creativecommons.org/licenses/by-nc/4.0/'})
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_frozen_before_acquisition':True,'frozen_artifact_sha256':artifact_hashes(),'source_revision':REVISION,'selection_protocol':'Before viewing: fixed take03 and take06 at20,45,90degree nominal releases, source2..10seconds,960x54030fps. All six attempted.','roi_protocol':'First decoded frame only: fixed HSV hue25..90 saturation>=65 value>=35; largest connected component of at least80pixels, bounding box padded6pixels, no later frame inspection or tuning. If no component, record failure.','authors':'Rasul Khanbayov, Mohamed Rayan Barhdadi, Erchin Serpedin, Hasan Kurban','records':records}
    PROVENANCE.parent.mkdir(parents=True,exist_ok=True);save(manifest)
    return manifest


def save(manifest): PROVENANCE.write_text(json.dumps(manifest,indent=2)+'\n')


def fetch(record):
    path=ROOT/record['path']
    if path.exists(): return
    path.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-ss','2','-i',record['source_url'],'-t','8','-vf','scale=960:540','-r','30','-an','-c:v','libx264','-preset','veryfast','-crf','20',str(path)],check=True,timeout=240)


def first_frame_roi(path):
    import cv2
    import numpy as np
    cap=cv2.VideoCapture(str(path));ok,frame=cap.read();cap.release()
    if not ok: raise ValueError('First frame did not decode.')
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,0]>=25)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>=65)&(hsv[:,:,2]>=35)).astype('uint8')*255
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    choices=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=80]
    if not choices: raise ValueError('No green component found by frozen first-frame ROI rule.')
    selected=max(choices,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    x,y,w,h,area=map(int,stats[selected]);left=max(0,x-6);top=max(0,y-6)
    right=min(frame.shape[1],x+w+6);bottom=min(frame.shape[0],y+h+6)
    return [left,top,right-left,bottom-top],{'green_component_area_px':area,'center':centers[selected].tolist(),'first_frame_only':True}


def evaluate(manifest,output):
    import numpy as np
    import torch
    from realityforge.tracking import track_video
    from realityforge.inference import infer_track
    current=artifact_hashes()
    if current!=manifest['frozen_artifact_sha256']:
        changed=[name for name in current if current[name]!=manifest['frozen_artifact_sha256'].get(name)]
        raise RuntimeError('Frozen estimator changed; cannot silently call this the same blind test: '+', '.join(changed))
    rows=[]
    for record in manifest['records']:
        row={'id':record['id'],'source_path':record['source_path'],'methods':{},'expected_model':'pendulum','status':'pending'}
        path=ROOT/record['path']
        if record.get('acquisition_error'):
            row.update(status='acquisition_failed',error=record['acquisition_error']);rows.append(row);continue
        try:
            if not path.exists(): raise FileNotFoundError('Missing clip. Run with --fetch to acquire public CC BY-NC data.')
            actual=digest(path)
            if record.get('sha256') and record['sha256']!=actual: raise ValueError('Recording hash mismatch; preserve exact delivered clips.')
            record['sha256']=actual;record['bytes']=path.stat().st_size
            if 'roi' not in record:
                roi,selection=first_frame_roi(path)
                record.update(roi=roi,roi_evidence=selection);save(manifest)
            settings={'mode':'pendulum','roi':record['roi'],'fit_fraction':.65}
            start=time.perf_counter();track=track_video(path,record['roi'],fit_fraction=.65)
            row.update(tracking_runtime_s=time.perf_counter()-start,roi=record['roi'],tracking_mean_confidence=float(track.confidence.mean()),tracking_low_confidence_fraction=float(np.mean(track.confidence<.4)),frames=track.frames,tracking_horizon_s=float(track.time[-1]),status='evaluated')
            for method in ('hybrid','analytical_refine','iris_recurrence'):
                start=time.perf_counter()
                try:
                    result=infer_track(track,settings,method=method)
                    p=result['simulation']['parameters'];length=9.80665/p['k']
                    row['methods'][method]={'parameters':p,'fit_rmse_px':result['metrics']['fit_rmse_px'],'forecast_rmse_px':result['metrics']['forecast_rmse_px'],'inference_runtime_s':time.perf_counter()-start,'effective_length_assuming_earth_g_m':length,'nominal_rope_length_m':.5,'relative_difference_to_nominal_rope_length':abs(length-.5)/.5,'warnings':result['warnings'],'fit_count':result['fit_count'],'fit_end_s':result['fit_end']}
                except Exception as error:
                    row['methods'][method]={'error':f'{type(error).__name__}: {error}','inference_runtime_s':time.perf_counter()-start}
            try:
                auto=infer_track(track,{**settings,'mode':'auto'},method='analytical')
                row['automatic_selection']={'selected':auto['model'],'diagnostics':auto['selection']}
            except Exception as error:row['automatic_selection']={'selected':'abstain','error':str(error)}
        except Exception as error:row.update(status='failed',error=f'{type(error).__name__}: {error}')
        rows.append(row);print(record['id'],json.dumps(row),flush=True)
    summaries={}
    for method in ('hybrid','analytical_refine','iris_recurrence'):
        values=[r['methods'][method] for r in rows if method in r['methods'] and 'error' not in r['methods'][method]]
        summaries[method]={'attempted':len(rows),'succeeded':len(values),'failed':len(rows)-len(values)}
        if values:
            for key in ('forecast_rmse_px','fit_rmse_px','inference_runtime_s','relative_difference_to_nominal_rope_length'):
                nums=[v[key] for v in values];summaries[method][key+'_median']=float(np.median(nums));summaries[method][key+'_max']=float(max(nums))
    report={'evaluation_script_sha256':digest(Path(__file__)), 'protocol_sha256':manifest.get('protocol_sha256'), 'protocol':'Frozen estimator before acquiring unseen take03/take06 at three IRIS pendulum release settings. Source2..10s, initial65% physical/geometry fitting, remaining35% forecast. First-frame-only fixed ROI rule. All six attempted; no hyperparameter tuning on these clips.','scope':'Held-out recording transfer relative to RealityForge development clips, not official IRIS split or independent manual tracking truth. Known dynamics family specified for parameter comparison; automatic selection evaluated separately.','limitations':['Errors are against automatically extracted image tracks.','Nominal rope length0.5m is not exact pivot-to-bob-center effective length.','No damping ground truth or calibrated uncertainty coverage for these recordings.','Published recurrence comparator adapts observed angles; it is not full EndPhys reproduction.','Network trained only on synthetic data; no IRIS frames or parameters used in training.'],'frozen_artifact_sha256':current,'manifest':'samples/blind-provenance.json','environment':{'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},'summary':summaries,'rows':rows}
    save(manifest);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(summaries,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fetch',action='store_true',help='Download separately licensed CC BY-NC4.0 IRIS clips');parser.add_argument('--initialize-only',action='store_true');parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/blind-iris.json');args=parser.parse_args()
    manifest=initialize()
    if args.initialize_only: return
    if args.fetch:
        for record in manifest['records']:
            try:
                fetch(record)
                record.pop('acquisition_error',None)
            except Exception as error:record['acquisition_error']=f'{type(error).__name__}: {error}'
            save(manifest)
    manifest.setdefault('first_evaluation_script_sha256',manifest.get('evaluation_script_sha256',digest(Path(__file__))))
    manifest['evaluation_script_sha256']=digest(Path(__file__))
    save(manifest)
    evaluate(manifest,args.output)

if __name__=='__main__':main()
