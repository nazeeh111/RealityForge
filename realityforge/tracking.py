"""CPU video tracking with inspectable template and foreground modes."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class Track:
    time: np.ndarray
    xy: np.ndarray
    confidence: np.ndarray
    width: int
    height: int
    fps: float
    frames: int
    warnings: list


def video_info(path, max_frames=1800, max_side=1280):
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    if not np.isfinite(fps) or not 1<=fps<=240:
        raise ValueError('Video requires a valid frame rate between 1 and 240 fps.')
    if width<2 or height<2 or source_frames<16:
        raise ValueError('At least 16 decoded frames and a valid video size are required.')
    stride=max(1,int(np.ceil(fps/30)))
    count=min(max_frames,int(np.floor(60*fps/stride))+1,int(np.ceil(source_frames/stride)))
    scale=min(1.,max_side/max(width,height))
    return fps,width,height,stride,count,scale


def iter_video(path, info):
    fps,width,height,stride,count,scale=info
    cap=cv2.VideoCapture(str(path));i=0;kept=0
    try:
        while kept<count:
            ok,frame=cap.read()
            if not ok: break
            if i%stride==0:
                yield i/fps,cv2.resize(frame,(round(width*scale),round(height*scale)))
                kept+=1
            i+=1
    finally: cap.release()


def track_video(path, roi=None, fit_fraction=.65):
    if not .3<=float(fit_fraction)<=.85: raise ValueError('Fit fraction must be between 0.30 and 0.85.')
    info=video_info(path)
    fps,width,height,stride,count,scale=info
    # Two passes bound memory: at most 31 grayscale frames for background,
    # then one current color frame while tracking. Future pixels never enter the background.
    iterator=iter_video(path,info)
    try: _,first=next(iterator)
    except StopIteration: raise ValueError('At least 16 decoded frames are required.')
    iterator.close()
    first_gray=cv2.cvtColor(first,cv2.COLOR_BGR2GRAY)
    background=None
    if roi is None:
        fit_count=max(1,int(count*fit_fraction))
        indices=set(np.linspace(0,fit_count-1,min(31,fit_count)).astype(int))
        backgrounds=[]
        for index,(_,frame) in enumerate(iter_video(path,info)):
            if index in indices: backgrounds.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
            if index>=fit_count-1: break
        background=np.median(backgrounds,axis=0).astype('uint8')
    warnings=[]
    if count==min(1800,int(np.floor(60*fps/stride))+1): warnings.append('Video truncated at the 60-second or 1800-frame analysis limit.')
    coords=[];scores=[];timestamps=[]
    if roi is not None:
        x,y,w,h=np.round(np.array(roi)*scale).astype(int)
        if w<5 or h<5 or x<0 or y<0 or x+w>first_gray.shape[1] or y+h>first_gray.shape[0]:
            raise ValueError('ROI must be a rectangle inside the first frame, at least 5 pixels wide and tall.')
        template=first_gray[y:y+h,x:x+w].copy()
        if float(template.std())<2: raise ValueError('Selected region is too uniform to track. Include an object edge.')
        previous=np.array([x+w/2,y+h/2],float);velocity=np.zeros(2)
        color_roi=cv2.cvtColor(first[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
        colorful=(color_roi[:,:,1]>75)&(color_roi[:,:,2]>35)
        use_color=float(colorful.mean())>.12
        hue=0
        if use_color:
            hist=np.bincount(color_roi[:,:,0][colorful],minlength=180)
            hue=int(hist.argmax())
        for timestamp,color_frame in iter_video(path,info):
            timestamps.append(timestamp)
            frame=cv2.cvtColor(color_frame,cv2.COLOR_BGR2GRAY)
            if use_color:
                hsv=cv2.cvtColor(color_frame,cv2.COLOR_BGR2HSV)
                delta=np.abs(hsv[:,:,0].astype(int)-hue)
                mask=(((np.minimum(delta,180-delta)<14)&(hsv[:,:,1]>65)&(hsv[:,:,2]>35)).astype('uint8'))*255
                count,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
                choices=[]
                for j in range(1,count):
                    area=stats[j,cv2.CC_STAT_AREA]
                    distance=np.linalg.norm(centers[j]-(previous+velocity))
                    if area>max(10,w*h*.025) and distance<max(150,max(w,h)*4):
                        choices.append((distance/100+abs(np.log(area/max(1,colorful.sum())))*.15,j))
                if choices:
                    j=min(choices)[1];point=centers[j]
                    velocity=.5*velocity+.5*(point-previous);previous=point
                    coords.append(point/scale);scores.append(float(min(1,stats[j,cv2.CC_STAT_AREA]/max(1,colorful.sum()))))
                    continue
            margin=max(45,int(max(w,h)*2.5))
            predicted=previous+velocity
            left=max(0,int(predicted[0]-w/2-margin));top=max(0,int(predicted[1]-h/2-margin))
            right=min(frame.shape[1],int(predicted[0]+w/2+margin));bottom=min(frame.shape[0],int(predicted[1]+h/2+margin))
            search=frame[top:bottom,left:right]
            if search.shape[0]<h or search.shape[1]<w:
                raise ValueError('Tracked object left the frame. Shorten the clip or choose another region.')
            result=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
            _,score,_,loc=cv2.minMaxLoc(result)
            point=np.array([left+loc[0]+w/2,top+loc[1]+h/2])
            velocity=.5*velocity+.5*(point-previous)
            previous=point
            coords.append(point/scale);scores.append(float(score))
    else:
        previous=None;velocity=np.zeros(2)
        for timestamp,color_frame in iter_video(path,info):
            timestamps.append(timestamp)
            frame=cv2.cvtColor(color_frame,cv2.COLOR_BGR2GRAY)
            diff=cv2.absdiff(frame,background)
            threshold=max(20,float(np.quantile(diff,.985)))
            mask=(diff>threshold).astype('uint8')*255
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
            candidates=[]
            for j in range(1,n):
                x,y,w,h,area=stats[j]
                if 12<area<frame.size*.08 and w>3 and h>3:
                    compact=area/(w*h)
                    dist=0 if previous is None else np.linalg.norm(centers[j]-(previous+velocity))
                    merit=np.sqrt(area)*compact/(1+dist/15)
                    candidates.append((merit,centers[j],compact))
            if not candidates:
                if previous is None: raise ValueError('No moving object detected. Select its bounding box in the first frame.')
                point=previous;score=0.
            else:
                _,point,score=max(candidates,key=lambda c:c[0])
            if previous is not None: velocity=.5*velocity+.5*(point-previous)
            previous=point
            coords.append(point/scale);scores.append(score)
    xy=np.array(coords); confidence=np.array(scores)
    if np.mean(confidence<.4)>.15: warnings.append('Tracking confidence is low in more than 15% of frames. Inspect the overlay and choose a tighter object region.')
    if np.ptp(xy,axis=0).max()<8: raise ValueError('Insufficient motion: tracked displacement is less than 8 pixels.')
    if len(xy)<16: raise ValueError('At least 16 decoded frames are required.')
    return Track(np.array(timestamps),xy,confidence,width,height,fps,len(xy),warnings)
