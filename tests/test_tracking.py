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
