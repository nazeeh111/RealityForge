"""Record repeatable release checks against the exact source/model/web bytes."""
import hashlib,json,platform,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'benchmarks/verification';DEST.mkdir(parents=True,exist_ok=True)
commands=[[sys.executable,'-m','pytest','-q'],[sys.executable,'scripts/smoke_installed.py']]
checks=[]
for index,command in enumerate(commands):
    start=time.perf_counter()
    process=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    output=process.stdout.replace(str(ROOT),'<repository>').replace(str(ROOT.parents[1]),'<task-workspace>')
    filename=['pytest.txt','pipeline-smoke.txt'][index];(DEST/filename).write_text(output)
    checks.append({'command':['python',*command[1:]],'exit_code':process.returncode,'runtime_s':time.perf_counter()-start,'log':filename})
    print(output,flush=True)
    if process.returncode:break
files=[p for directory in ('realityforge','tests','models','web') for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.pt','.pt2','.torchscript','.js','.html','.css','.svg')]
report={'utc_time':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'platform':platform.platform(),'checks':checks,'artifact_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
(DEST/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
sys.exit(any(c['exit_code']!=0 for c in checks))
