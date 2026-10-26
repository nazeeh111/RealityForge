"""Decoded-video tests with original generated pixels and known centroids."""
import cv2
import numpy as np
import pytest
from realityforge.tracking import track_video


def write_clip(path, altered_future=False):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(320,200))
    if not writer.isOpened(): pytest.skip('OpenCV MJPEG writer unavailable on this platform')
    truth=[]
    for index in range(40):
        point=(30+4*index,100)
        if altered_future and index>=26: point=(290,40)
        frame=np.full((200,320,3),220,dtype=np.uint8)
        cv2.circle(frame,point,8,(0,0,220),-1)
        writer.write(frame);truth.append(point)
    writer.release()
    return np.array(truth)


def test_color_roi_tracks_decoded_video(tmp_path):
    path=tmp_path/'original.avi';truth=write_clip(path)
    track=track_video(path,roi=[20,90,20,20])
    assert track.xy.shape==(40,2)
    assert np.sqrt(np.mean(np.sum((track.xy-truth)**2,axis=1)))<1
    assert track.time[-1]==pytest.approx(39/20)


def test_foreground_fit_tracking_is_independent_of_future_frames(tmp_path):
    before=tmp_path/'before.avi';after=tmp_path/'after.avi'
    write_clip(before);write_clip(after,altered_future=True)
    first=track_video(before,fit_fraction=.65)
    second=track_video(after,fit_fraction=.65)
    np.testing.assert_allclose(first.xy[:26],second.xy[:26],atol=.1)


def test_outside_roi_is_rejected(tmp_path):
    path=tmp_path/'original.avi';write_clip(path)
    with pytest.raises(ValueError,match='ROI'): track_video(path,roi=[310,100,30,30])


def test_long_video_is_bounded_in_memory_and_simulation_horizon(tmp_path):
    """A long compressed clip must not require retaining all decoded RGB frames."""
    import tracemalloc
    path=tmp_path/'long.avi';fps=25
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),fps,(320,200))
    if not writer.isOpened(): pytest.skip('OpenCV MJPEG writer unavailable on this platform')
    for index in range(70*fps):
        point=(int(160+100*np.sin(index/fps)),100)
        frame=np.full((200,320,3),220,dtype=np.uint8)
        cv2.circle(frame,point,8,(0,0,220),-1);writer.write(frame)
    writer.release()
    tracemalloc.start()
    try:
        track=track_video(path,roi=[150,90,20,20])
        _,peak=tracemalloc.get_traced_memory()
    finally: tracemalloc.stop()
    assert peak<25*1024*1024, f'Python/NumPy peak allocation was {peak/1024**2:.1f}MiB'
    assert track.time[-1]<=60
    assert any('truncat' in w.lower() for w in track.warnings)


def _write_ramp_clip(path, count=60, bright_future=False):
    """Lossless fixture whose suffix can change a wrongly sampled median."""
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),30,(160,120))
    if not writer.isOpened(): pytest.skip('OpenCV FFV1 writer unavailable on this platform')
    frames=[]
    try:
        for i in range(count):
            level=2*i if i<39 else (190 if bright_future else 0)
            frame=np.full((120,160,3),level,dtype=np.uint8)
            frame[48:56,12+2*i:20+2*i]=250
            frame[-1,0]=(i,255-i,3*i%256)
            writer.write(frame);frames.append(frame)
    finally: writer.release()
    cap=cv2.VideoCapture(str(path));decoded=[]
    try:
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==count
        while True:
            ok,frame=cap.read()
            if not ok: break
            decoded.append(frame)
    finally: cap.release()
    assert len(decoded)==count
    for expected,actual in zip(frames,decoded): np.testing.assert_array_equal(expected,actual)
    return decoded


def _capture_control(monkeypatch, metadata=None, stop_after=None):
    """Native decoding with explicitly injected metadata/EOF and release audit."""
    native=cv2.VideoCapture;opened=[];stop_after=stop_after or {}
    class Capture:
        def __init__(self,*args,**kwargs):
            self.cap=native(*args,**kwargs);self.ordinal=len(opened)
            self.successful=0;self.released=False;opened.append(self)
        def get(self,key):
            if metadata is not None and key==cv2.CAP_PROP_FRAME_COUNT: return metadata
            return self.cap.get(key)
        def read(self):
            if self.ordinal in stop_after and self.successful>=stop_after[self.ordinal]: return False,None
            result=self.cap.read();self.successful+=int(result[0]);return result
        def release(self):
            self.cap.release();self.released=True
    monkeypatch.setattr(cv2,'VideoCapture',Capture)
    return opened


@pytest.mark.parametrize('reported_count',[None,180])
def test_foreground_preprocessing_excludes_native_forecast_suffix(tmp_path,monkeypatch,reported_count):
    paths=[tmp_path/'dark.mkv',tmp_path/'bright.mkv']
    decoded=[_write_ramp_clip(path,bright_future=bool(i)) for i,path in enumerate(paths)]
    for a,b in zip(decoded[0][:39],decoded[1][:39]): np.testing.assert_array_equal(a,b)
    assert not np.array_equal(decoded[0][39],decoded[1][39])
    captures=_capture_control(monkeypatch,metadata=reported_count)
    median=np.median;backgrounds=[];selected=[];lookup={}
    def inspect_median(frames,*args,**kwargs):
        selected.append([lookup[frame.tobytes()] for frame in frames])
        result=median(frames,*args,**kwargs);backgrounds.append(result.astype('uint8'))
        return result
    monkeypatch.setattr(np,'median',inspect_median)
    tracks=[]
    for path,frames in zip(paths,decoded):
        lookup={cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).tobytes():i for i,frame in enumerate(frames)}
        tracks.append(track_video(path,fit_fraction=.65))
    assert all(track.frames==60 for track in tracks)
    # The accepted inference split for 60 frames at .65 is 39: indices 0..38.
    assert all(indices and max(indices)<39 and len(indices)<=31 for indices in selected)
    np.testing.assert_array_equal(backgrounds[0],backgrounds[1])
    gray=cv2.cvtColor(decoded[0][0],cv2.COLOR_BGR2GRAY)
    np.testing.assert_array_equal(cv2.absdiff(gray,backgrounds[0]),cv2.absdiff(gray,backgrounds[1]))
    np.testing.assert_array_equal(tracks[0].xy[:39],tracks[1].xy[:39])
    np.testing.assert_array_equal(tracks[0].confidence[:39],tracks[1].confidence[:39])
    assert all(cap.released for cap in captures)


def test_native_short_clip_rejects_overreported_metadata(tmp_path,monkeypatch):
    path=tmp_path/'short.mkv';_write_ramp_clip(path,count=15)
    captures=_capture_control(monkeypatch,metadata=180)
    with pytest.raises(ValueError,match='^At least 16 decoded frames are required\\.$'):
        track_video(path)
    assert all(cap.released for cap in captures)


# Capture opens: metadata=0, initial count=1, background=2, final tracking=3.
@pytest.mark.parametrize('stop_after,minimum_error',[
    ({1:0},True),({1:15},True),
    ({3:59},False),({3:0},False),({1:40},False),
    ({2:0},False),({2:20},False),
])
def test_native_decode_pass_changes_are_rejected(tmp_path,monkeypatch,stop_after,minimum_error):
    path=tmp_path/'stable.mkv';_write_ramp_clip(path)
    captures=_capture_control(monkeypatch,metadata=180,stop_after=stop_after)
    if 2 in stop_after:
        def unexpected_median(*args,**kwargs):
            pytest.fail('Incomplete background sampling must fail before median')
        monkeypatch.setattr(np,'median',unexpected_median)
    message=('At least 16 decoded frames are required.' if minimum_error else
             'Video decoded frame count changed between passes. Retry with a stable local clip.')
    with pytest.raises(ValueError) as error: track_video(path)
    assert str(error.value)==message
    assert all(cap.released for cap in captures)


@pytest.mark.parametrize('roi',[None,[20,90,20,20]])
@pytest.mark.parametrize('reported_count',[None,180])
def test_nominal_and_roi_preserve_known_native_motion(tmp_path,monkeypatch,roi,reported_count):
    path=tmp_path/'known.avi';truth=write_clip(path)
    captures=_capture_control(monkeypatch,metadata=reported_count)
    track=track_video(path,roi=roi)
    assert track.frames==40
    np.testing.assert_allclose(track.xy,truth,atol=1)
    np.testing.assert_array_equal(track.time,np.arange(40)/20)
    assert track.warnings==[]
    assert np.all(track.confidence>.4)
    assert all(cap.released for cap in captures)
