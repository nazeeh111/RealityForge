"""Caption actual browser footage without altering its timing or model results."""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main(recording,output):
    report=json.loads(Path(recording).read_text());raw=Path(report['raw_video']);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(raw)]))
    duration=float(metadata['format']['duration']);events={e['name']:e['time_s'] for e in report['events']}
    captions=[(0,events['recovery_requested'],'01 / Choose an experiment','Original generated fixture. Real recording tests are reported separately.'),(events['recovery_requested'],events['recovery_completed'],'02 / Recover dynamics on the local CPU','Actual tracking, neural initialization and differentiable physical fitting.'),(events['recovery_completed'],events['real_counterfactual_ready'],'03 / Compare observation and held-out forecast','Only the first 65% of motion fits the model.'),(events['real_counterfactual_ready'],events['heldout_comparison'],'04 / Change gravity / length','Orange: changed conditions. Blue: recovered physical model.'),(events['heldout_comparison'],duration,'05 / Inspect the forecast and export the evidence','Trajectory, parameters, local uncertainty and provenance.')]
    font=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    if not font.exists():font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    with tempfile.TemporaryDirectory(prefix='realityforge-demo-') as temp:
        filters=['drawbox=x=0:y=1012:w=iw:h=68:color=0x243132:t=fill']
        for i,(start,end,title,subtitle) in enumerate(captions):
            for j,(text,size,y,color) in enumerate([(title,19,1023,'white'),(subtitle,13,1052,'0xb9cbbd')]):
                path=Path(temp)/f'{i}-{j}.txt';path.write_text(text)
                filters.append(f"drawtext=expansion=none:fontfile='{font}':textfile='{path}':fontsize={size}:fontcolor={color}:x=34:y={y}:enable='between(t,{start},{end})'")
        mp4=out/'realityforge-demo.next.mp4'
        subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(raw),'-vf',','.join(filters),'-c:v','libx264','-pix_fmt','yuv420p','-crf','20','-movflags','+faststart','-an',str(mp4)],check=True)
    gif=out/'realityforge-demo.next.gif'
    start=events['real_counterfactual_ready']+.25
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-ss',str(start),'-t','5.8','-i',str(mp4),'-filter_complex','fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer','-loop','0',str(gif)],check=True)
    mp4.replace(out/'realityforge-demo.mp4');gif.replace(out/'realityforge-demo.gif')
    mp4=out/'realityforge-demo.mp4';gif=out/'realityforge-demo.gif'
    report.update(raw_video=raw.name,raw_video_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),duration_s=duration,editing='Added captions only; full original recording timing retained. GIF is a 5.8 s excerpt during actual counterfactual playback.',captions=[{'start_s':a,'end_s':b,'title':c,'subtitle':d} for a,b,c,d in captions],artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (mp4,gif)})
    (out/'demo-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'duration_s':duration,'mp4_bytes':mp4.stat().st_size,'gif_bytes':gif.stat().st_size},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',required=True);p.add_argument('--output',default=str(ROOT/'docs/assets'));a=p.parse_args();main(a.recording,a.output)
