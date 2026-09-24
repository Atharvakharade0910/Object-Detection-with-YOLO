"""Integration checks using the real YOLO weights and video encoder."""
import io
import time
import cv2
import pytest
import server

@pytest.fixture
def client():
    client = server.app.test_client()
    with client.session_transaction() as session:
        session['user'] = 'admin'
    return client

def image_form(**extra):
    return {'file': (io.BytesIO((server.ROOT/'images/office_4.jpg').read_bytes()), 'scene.jpg'), **extra}

def test_auth_and_origin_boundary(client):
    anonymous = server.app.test_client()
    assert anonymous.post('/api/detect').status_code == 401
    assert anonymous.get('/api/meta').status_code == 401
    assert anonymous.post('/api/login', json={'username':'admin','password':'incorrect'}).status_code == 401
    assert client.post('/api/logout', headers={'Origin':'https://untrusted.example'}).status_code == 403
    assert client.get('/api/health').json['frontend'] == 'HTML / CSS / JavaScript'
    assert b'/static/app.js' in client.get('/').data

def test_source_tabs_support_keyboard_navigation(client):
    page = client.get('/').get_data(as_text=True)
    script = (server.ROOT/'static/app.js').read_text(encoding='utf-8')
    assert 'role="tablist"' in page
    assert page.count('tabindex="-1" data-source=') == 2
    assert 'button.tabIndex=active?0:-1' in script
    for key in ('ArrowLeft', 'ArrowRight', 'Home', 'End'):
        assert f"event.key==='{key}'" in script

def test_inspector_tabs_support_keyboard_navigation(client):
    page = client.get('/').get_data(as_text=True)
    script = (server.ROOT/'static/app.js').read_text(encoding='utf-8')
    assert 'role="tablist" aria-label="Result view"' in page
    assert page.count('tabindex="-1" data-inspector=') == 1
    assert "document.querySelectorAll('[data-inspector]')" in script
    assert 'button.tabIndex=active?0:-1' in script
    for key in ('ArrowLeft', 'ArrowRight', 'Home', 'End'):
        assert script.count(f"event.key==='{key}'") == 2

def test_upload_hints_disclose_every_supported_format(client):
    page = client.get('/').get_data(as_text=True)
    script = (server.ROOT/'static/app.js').read_text(encoding='utf-8')
    assert 'JPG, PNG, WEBP, BMP · up to 100 MB' in page
    assert 'JPG, PNG, WEBP, BMP · up to 100 MB' in script
    assert 'MP4, MOV, AVI, MKV, WEBM · 100 MB / 2 min' in script

def test_real_detection_and_class_filter(client):
    response = client.post('/api/detect', data=image_form())
    assert response.status_code == 200
    result = response.json
    assert result['detections'] and result['width'] > 0
    assert sum(result['counts'].values()) == len(result['detections'])
    assert result['image'].startswith('data:image/jpeg;base64,')
    selected = result['detections'][0]['class_id']
    filtered = client.post('/api/detect', data=image_form(classes=f'[{selected}]')).json
    assert filtered['detections']
    assert all(d['class_id'] == selected for d in filtered['detections'])
    assert len(client.get('/api/meta').json['classes']) == 80

def test_real_segmentation(client):
    response = client.post('/api/detect', data=image_form(task='segment'))
    assert response.status_code == 200
    assert response.json['task'] == 'segment'
    assert any(len(d.get('polygon', [])) >= 3 for d in response.json['detections'])

@pytest.mark.parametrize('extra', [{'confidence':'nan'},{'confidence':'1.5'},{'classes':'[999]'},{'classes':'null'},{'task':'unknown'}])
def test_invalid_options(client, extra):
    assert client.post('/api/detect', data=image_form(**extra)).status_code == 400

def test_invalid_media(client):
    assert client.post('/api/detect', data={'file':(io.BytesIO(b'not an image'),'bad.jpg')}).status_code == 400
    assert client.post('/api/videos', data={'file':(io.BytesIO(b'bad'),'bad.exe')}).status_code == 400

def test_video_tracking_encoding_and_ownership(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'OUTPUT', tmp_path)
    monkeypatch.setattr(server, 'jobs', {})
    frame = cv2.imread(str(server.ROOT/'images/office_4.jpg'))
    frame = cv2.resize(frame, (640, 480))
    source = tmp_path/'test.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 4, (640,480))
    assert writer.isOpened()
    for _ in range(4): writer.write(frame)
    writer.release()
    response = client.post('/api/videos', data={'file':(io.BytesIO(source.read_bytes()),'test.mp4'),'tracking':'true'})
    assert response.status_code == 202
    job_id = response.json['id']
    assert server.app.test_client().get(f'/api/videos/{job_id}').status_code == 401
    other = server.app.test_client()
    with other.session_transaction() as session: session['user'] = 'another-user'
    assert other.get(f'/api/videos/{job_id}').status_code == 404
    deadline = time.monotonic()+60
    while time.monotonic() < deadline:
        status = client.get(f'/api/videos/{job_id}').json
        if status['status'] in ('complete','failed'): break
        time.sleep(.1)
    assert status['status'] == 'complete', status
    assert status['frames'] == 4 and status['counts'] and status['unique_tracks'] > 0
    response = client.get(f'/api/videos/{job_id}/file')
    assert response.status_code == 200 and response.mimetype == 'video/mp4'
    assert len(response.data) > 1000
    response.close()
    assert other.get(f'/api/videos/{job_id}/file').status_code == 404

def test_logout(client):
    assert client.post('/api/logout').status_code == 200
    assert client.get('/api/meta').status_code == 401
