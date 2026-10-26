import argparse
import json
from .inference import analyze


def main():
    p=argparse.ArgumentParser(description='RealityForge planar video-to-physics')
    sub=p.add_subparsers(dest='command',required=True)
    serve=sub.add_parser('serve');serve.add_argument('--port',type=int,default=8000)
    fit=sub.add_parser('analyze');fit.add_argument('video');fit.add_argument('--settings',default='{}');fit.add_argument('--output',default='analysis.json')
    a=p.parse_args()
    if a.command=='serve':
        import uvicorn
        uvicorn.run('realityforge.api:app',host='127.0.0.1',port=a.port)
    else:
        result=analyze(a.video,json.loads(a.settings))
        with open(a.output,'w') as f: json.dump(result,f,indent=2,allow_nan=False)
        print(json.dumps(result['metrics'],indent=2))

if __name__=='__main__': main()
