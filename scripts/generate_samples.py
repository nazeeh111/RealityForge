"""Original rendered fixtures, not experimental validation."""
import json, subprocess
from pathlib import Path
import cv2
import numpy as np
from realityforge.physics import simulate
ROOT=Path(__file__).resolve().parents[1]

def main():
    root=ROOT/'samples'/'generated';root.mkdir(parents=True,exist_ok=True);samples=[]
    for mode in ('pendulum','sliding'):
        t=np.arange(0,6,1/30)
        if mode=='pendulum':
            p={'k':17.5,'beta':.16,'theta0':.62,'omega0':0.};geo={'pivot':[480,105],'radius':280}
            settings={'mode':mode,'pivot':[480,105],'fit_fraction':.65,'meters_per_pixel':.002}
        else:
            p={'acceleration':-21.,'v0':135.,'x0':0.};geo={'origin':[135,310],'direction':[1,0]}
            settings={'mode':mode,'roi':[111,285,48,50],'fit_fraction':.65,'meters_per_pixel':.005}
        xy=simulate(mode,p,geo,t);avi=root/f'{mode}.avi'
        writer=cv2.VideoWriter(str(avi),cv2.VideoWriter_fourcc(*'MJPG'),30,(960,540))
        for i,point in enumerate(xy):
            frame=np.empty((540,960,3),np.uint8);frame[:]=[234,239,241]
            for gx in range(0,960,40): cv2.line(frame,(gx,70),(gx,470),(222,228,229),1)
            for gy in range(70,471,40): cv2.line(frame,(0,gy),(960,gy),(222,228,229),1)
            cv2.putText(frame,'REALITYFORGE / GENERATED FIXTURE',(34,38),cv2.FONT_HERSHEY_SIMPLEX,.65,(85,96,97),1,cv2.LINE_AA)
            q=tuple(np.round(point).astype(int))
            if mode=='pendulum':
                cv2.line(frame,(480,105),q,(87,100,103),3,cv2.LINE_AA);cv2.circle(frame,(480,105),7,(59,78,81),-1)
                cv2.circle(frame,q,19,(54,104,210),-1,cv2.LINE_AA);cv2.circle(frame,(q[0]-5,q[1]-6),5,(144,183,241),-1)
                if i==0: settings['roi']=[int(q[0])-23,int(q[1])-23,46,46]
            else:
                cv2.line(frame,(85,338),(875,338),(85,100,100),3);cv2.rectangle(frame,(q[0]-21,q[1]-22),(q[0]+21,q[1]+22),(100,117,47),-1)
                cv2.line(frame,(q[0]-12,q[1]-15),(q[0]+12,q[1]+14),(194,215,152),3)
            cv2.putText(frame,f'{i/30:05.2f} s',(840,505),cv2.FONT_HERSHEY_SIMPLEX,.65,(85,96,97),1,cv2.LINE_AA);writer.write(frame)
        writer.release();output=root/f'{mode}.mp4'
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(avi),'-c:v','libx264','-pix_fmt','yuv420p','-crf','22',str(output)],check=True);avi.unlink()
        (root/f'{mode}-truth.json').write_text(json.dumps({'kind':'synthetic','parameters':p,'geometry':geo,'time':t.tolist(),'xy':xy.tolist(),'seed':31},indent=2))
        samples.append({'id':f'generated-{mode}','title':f'{mode.title()} / generated','description':'Original rendered fixture with known ground truth. Synthetic, not experimental evidence.','kind':'synthetic','video_url':f'/samples/generated/{mode}.mp4','settings':settings})
    manifest=ROOT/'samples'/'manifest.json';existing=json.loads(manifest.read_text())['samples'] if manifest.exists() else []
    manifest.write_text(json.dumps({'samples':samples+[s for s in existing if not s['id'].startswith('generated-')]},indent=2)+'\n')
if __name__=='__main__': main()
