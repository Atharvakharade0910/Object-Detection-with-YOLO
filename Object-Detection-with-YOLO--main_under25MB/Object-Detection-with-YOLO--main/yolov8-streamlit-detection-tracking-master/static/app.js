'use strict';
const $ = id => document.getElementById(id);
const state = {user:null, source:'image', file:null, image:null, result:null, classes:[], selectedClasses:new Set(), selectedObject:null, inspector:'objects', busy:false, live:false, stream:null, job:null, sourceUrl:null, generation:0, ready:false};
const palette = ['#c7f56b','#73d8f4','#ffbc77','#eaa3fa','#ff9299','#a2baff','#79dfbd'];
let toastTimer;
function notify(message, error=false){
  $('statusMessage').textContent=message; $('statusMessage').classList.toggle('error',error); $('statusMessage').hidden=false;
  clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('statusMessage').hidden=true,error?8000:4500);
}
async function api(url, options={}){
  const controller=new AbortController(), timeout=setTimeout(()=>controller.abort(),120000);
  try{
    const response=await fetch(url,{...options,signal:controller.signal});
    const data=await response.json();
    if(!response.ok){
      if(response.status===401 && url!=='/api/login'){
        stopCamera(); state.user=null; state.ready=false;
        if(!$('loginDialog').open) $('loginDialog').showModal();
      }
      throw new Error(data.error||'The request could not be completed.');
    }
    return data;
  }catch(error){
    if(error.name==='AbortError') throw new Error('The request timed out. Please try again.');
    throw error;
  }finally{clearTimeout(timeout);}
}
const jsonRequest=(method,data)=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
function controls(){
  const locked=state.busy||state.live||!!state.job;
  document.querySelectorAll('[data-source],#uploadZone,#sampleButton,#emptySampleButton,#taskSelect,#confidence,#classSearch,#clearClasses,#tracking,.class-chip').forEach(el=>el.disabled=locked);
  $('cameraButton').disabled=state.busy||!!state.job;
  $('runButton').disabled=!state.ready||state.busy||!!state.job||(state.source==='webcam'?!state.stream:!state.file);
  $('runLabel').textContent=state.live?'Stop live detection':state.source==='webcam'?'Start live detection':state.source==='video'?'Analyze video':$('taskSelect').value==='segment'?'Run segmentation':'Run detection';
  $('busyOverlay').hidden=!state.busy||state.live;
}
function resetResults(){
  state.result=null; state.selectedObject=null;
  ['objectCount','classCount','inferenceTime'].forEach(id=>$(id).textContent='—');
  $('countCaption').textContent='Waiting for a scene'; $('timeCaption').textContent='Measured on this machine';
  $('resultBadge').textContent='0'; $('inspectorEmpty').hidden=false;
  $('inspectorEmpty').querySelector('h3').textContent='There’s more to see.';
  $('inspectorEmpty').querySelector('p').textContent='Run an analysis to reveal object classes, confidence scores, and scene details.';
  $('detectionList').replaceChildren(); $('summaryList').replaceChildren();
  $('detectionList').hidden=true; $('summaryList').hidden=true;
  ['downloadImage','downloadJson','downloadCsv','printReport'].forEach(id=>$(id).disabled=true);
  $('downloadImage').textContent='Download image ↓';
}
function stopLive(){state.live=false;state.generation++;controls();}
function stopCamera(){
  stopLive();
  state.stream?.getTracks().forEach(track=>track.stop()); state.stream=null;
  $('cameraPreview').srcObject=null; $('cameraButton').textContent='Enable camera'; controls();
}
function clearMedia(){
  if(state.sourceUrl){URL.revokeObjectURL(state.sourceUrl);state.sourceUrl=null;}
  $('videoPreview').pause(); $('videoPreview').removeAttribute('src'); $('videoPreview').load();
  ['previewCanvas','videoPreview','cameraPreview','previewFilename'].forEach(id=>$(id).hidden=true);
  $('emptyState').hidden=false; $('previewInfo').textContent='No source selected';
  state.file=null;state.image=null;
}
function changeSource(source){
  if(state.busy||state.job||state.live) return;
  stopCamera();clearMedia();resetResults();state.source=source;state.generation++;
  document.querySelectorAll('[data-source]').forEach(button=>{const active=button.dataset.source===source;button.classList.toggle('active',active);button.setAttribute('aria-selected',active);});
  $('sourceBadge').textContent=source.toUpperCase(); $('uploadZone').hidden=source==='webcam';
  $('cameraButton').hidden=source!=='webcam'; $('sampleButton').hidden=source!=='image';
  $('trackingRow').hidden=source!=='video';$('jobPanel').hidden=true;
  $('uploadTitle').textContent=source==='video'?'Drop a video here':'Drop an image here';
  $('uploadTypes').textContent=source==='video'?'MP4, MOV, AVI, WEBM · 100 MB / 2 min':'JPG, PNG, WEBP · up to 100 MB';
  $('fileInput').accept=source==='video'?'.mp4,.mov,.avi,.mkv,.webm':'image/jpeg,image/png,image/webp,image/bmp';
  $('fileInput').value=''; document.querySelector('.overlay-controls').hidden=source==='video'; controls();
}
function loadImage(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('The image could not be opened. Try a JPG or PNG.'));image.src=url;});}
async function acceptFile(file){
  if(!file||state.busy||state.job||state.live) return;
  if(file.size>100*1024*1024){notify('Choose a file under 100 MB.',true);return;}
  const isVideo=state.source==='video';
  if(!(isVideo?/\.(mp4|mov|avi|mkv|webm)$/i:/\.(jpe?g|png|webp|bmp)$/i).test(file.name)){notify(isVideo?'Choose a supported video file.':'Choose a JPG, PNG, WEBP or BMP image.',true);return;}
  clearMedia(); resetResults();state.busy=true;controls();
  try{
    state.sourceUrl=URL.createObjectURL(file);
    if(isVideo){$('videoPreview').src=state.sourceUrl;$('videoPreview').hidden=false;}
    else{state.image=await loadImage(state.sourceUrl);draw();$('previewCanvas').hidden=false;}
    state.file=file;$('emptyState').hidden=true;$('previewFilename').hidden=false;$('previewFilename').textContent=file.name;
    $('uploadTitle').textContent=file.name;
    $('previewInfo').textContent=isVideo?'Video ready · up to 2 minutes':`${state.image.naturalWidth} × ${state.image.naturalHeight} px · ready`;
  }catch(error){clearMedia();notify(error.message,true);}finally{state.busy=false;controls();}
}
async function sample(){
  if(state.busy||state.job||state.live)return;
  try{
    if(state.source!=='image')changeSource('image');
    const response=await fetch('/api/sample');if(!response.ok)throw new Error('Sign in to load the sample.');
    await acceptFile(new File([await response.blob()],'office-scene.jpg',{type:'image/jpeg'}));
    notify('Sample loaded. Run detection to explore this scene.');
  }catch(error){notify(error.message,true);}
}
function renderClasses(){
  const search=$('classSearch').value.toLowerCase(); const fragment=document.createDocumentFragment();
  const popular=['person','car','chair','laptop','cell phone','dog','bottle'];
  const sorted=[...state.classes].sort((a,b)=>(popular.includes(b.name)?1:0)-(popular.includes(a.name)?1:0)||a.id-b.id);
  sorted.filter(c=>c.name.includes(search)).forEach(c=>{
    const button=document.createElement('button');button.className='class-chip';button.textContent=c.name;
    button.classList.toggle('selected',state.selectedClasses.has(c.id));button.setAttribute('aria-pressed',state.selectedClasses.has(c.id));
    button.onclick=()=>{if(state.selectedClasses.has(c.id))state.selectedClasses.delete(c.id);else state.selectedClasses.add(c.id);renderClasses();markSettingsChanged();};fragment.append(button);
  });
  if(!fragment.childNodes.length){const empty=document.createElement('p');empty.className='field-help';empty.textContent='No matching classes.';fragment.append(empty);}
  $('classList').replaceChildren(fragment);
  $('classHint').textContent=state.selectedClasses.size?`${state.selectedClasses.size} classes selected. Reset to include all.`:'All classes included. Select to narrow your search.';
  controls();
}
function markSettingsChanged(){if(state.result)$('previewInfo').textContent='Settings changed · run again to update results';}
function draw(){
  if(!state.image)return;
  const canvas=$('previewCanvas'),ctx=canvas.getContext('2d');
  canvas.width=state.image.naturalWidth;canvas.height=state.image.naturalHeight;
  ctx.drawImage(state.image,0,0);
  const scale=Math.max(1,canvas.width/800),fontSize=Math.round(12*scale);
  for(const detection of state.result?.detections||[]){
    const color=palette[detection.class_id%palette.length],active=state.selectedObject===null||state.selectedObject===detection.id;
    ctx.save();ctx.globalAlpha=active?1:.22;
    if($('showMasks').checked&&detection.polygon?.length){
      ctx.beginPath();detection.polygon.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fillStyle=color;ctx.globalAlpha=active?.3:.08;ctx.fill();ctx.globalAlpha=active?1:.22;
    }
    const [x1,y1,x2,y2]=detection.box;
    if($('showBoxes').checked){ctx.strokeStyle=color;ctx.lineWidth=(state.selectedObject===detection.id?4:2)*scale;ctx.strokeRect(x1,y1,x2-x1,y2-y1);}
    if($('showLabels').checked){
      ctx.font=`600 ${fontSize}px "Segoe UI", sans-serif`;
      const label=`${detection.name}  ${Math.round(detection.confidence*100)}%`,width=ctx.measureText(label).width+12*scale;
      const x=Math.min(x1,Math.max(0,canvas.width-width)),y=Math.max(fontSize+9*scale,y1);
      ctx.fillStyle=color;ctx.fillRect(x,y-fontSize-8*scale,width,fontSize+8*scale);ctx.fillStyle='#10200b';ctx.fillText(label,x+6*scale,y-5*scale);
    }
    ctx.restore();
  }
}
function inspectorTab(tab){
  state.inspector=tab;
  document.querySelectorAll('[data-inspector]').forEach(button=>button.classList.toggle('active',button.dataset.inspector===tab));
  const populated=!!state.result&&Object.keys(state.result.counts||{}).length>0;
  $('detectionList').hidden=!populated||tab!=='objects'||!state.result?.detections;
  $('summaryList').hidden=!populated||tab!=='summary';
  if(populated&&tab==='objects'&&!state.result.detections){$('summaryList').hidden=false;}
}
function renderResults(){
  const result=state.result;if(!result)return;
  const count=result.detections?result.detections.length:Object.values(result.counts).reduce((a,b)=>a+b,0);
  $('objectCount').textContent=count.toLocaleString();$('classCount').textContent=Object.keys(result.counts).length;
  $('inferenceTime').textContent=result.elapsed_ms>=1000?`${(result.elapsed_ms/1000).toFixed(1)}s`:`${result.elapsed_ms}ms`;
  $('countCaption').textContent=result.detections?'In the current frame':'Detections across all frames';
  $('timeCaption').textContent=result.detections?'Model + queue time':'Total video processing time';
  $('resultBadge').textContent=count.toLocaleString();$('inspectorEmpty').hidden=count>0;
  if(!count){$('inspectorEmpty').querySelector('h3').textContent='No objects at this threshold.';$('inspectorEmpty').querySelector('p').textContent='Try a lower confidence threshold, reset the class filter, or choose a clearer scene.';}
  const fragment=document.createDocumentFragment();
  for(const detection of result.detections||[]){
    const button=document.createElement('button');button.className='detection-item';button.dataset.object=detection.id;
    button.setAttribute('aria-pressed',state.selectedObject===detection.id);button.classList.toggle('selected',state.selectedObject===detection.id);
    const dot=document.createElement('span');dot.className='object-dot';dot.style.background=palette[detection.class_id%palette.length];
    const text=document.createElement('span');text.className='object-text';
    const name=document.createElement('strong');name.textContent=detection.name;
    const dimensions=document.createElement('small');dimensions.textContent=`#${String(detection.id+1).padStart(2,'0')} · ${Math.round(detection.box[2]-detection.box[0])} × ${Math.round(detection.box[3]-detection.box[1])} px`;
    text.append(name,dimensions);const score=document.createElement('span');score.className='object-score';score.textContent=`${Math.round(detection.confidence*100)}%`;
    button.append(dot,text,score);button.onclick=()=>selectObject(detection.id);fragment.append(button);
  }
  $('detectionList').replaceChildren(fragment);$('summaryList').replaceChildren();
  for(const [name,value] of Object.entries(result.counts).sort((a,b)=>b[1]-a[1])){
    const row=document.createElement('div');row.className='summary-row';const heading=document.createElement('div');heading.className='summary-heading';
    const label=document.createElement('span');label.textContent=name;const number=document.createElement('span');number.textContent=value;heading.append(label,number);
    const bar=document.createElement('div');bar.className='summary-bar';const fill=document.createElement('span');fill.style.width=`${value/Math.max(1,count)*100}%`;bar.append(fill);row.append(heading,bar);$('summaryList').append(row);
  }
  const note=document.createElement('p');note.className='summary-note';
  note.textContent=result.detections?'Counts refer to this image or the latest camera frame. Model predictions may miss or misidentify objects.':`Counts are frame detections, not distinct real-world objects. ${result.tracking?`${result.unique_tracks} tracker IDs assigned. IDs can change after occlusion.`:'Tracking was disabled.'} Exported video has no audio.`;
  $('summaryList').append(note);
  ['downloadImage','downloadJson','downloadCsv','printReport'].forEach(id=>$(id).disabled=false);
  $('downloadImage').textContent=result.url?'Download video ↓':'Download image ↓';
  inspectorTab(state.inspector);
}
function selectObject(id){
  state.selectedObject=state.selectedObject===id?null:id;
  document.querySelectorAll('[data-object]').forEach(el=>{const selected=Number(el.dataset.object)===state.selectedObject;el.classList.toggle('selected',selected);el.setAttribute('aria-pressed',selected);});draw();
}
function formFor(file){
  const form=new FormData();form.append('file',file);form.append('task',$('taskSelect').value);form.append('confidence',Number($('confidence').value)/100);form.append('classes',JSON.stringify([...state.selectedClasses]));form.append('tracking',$('tracking').checked);return form;
}
async function applyResult(result,generation){
  const image=await loadImage(result.image);
  if(generation!==state.generation)return;
  state.image=image;state.result={...result,filename:state.file?.name||'camera-frame.jpg',threshold:Number($('confidence').value)/100,selected_classes:[...state.selectedClasses],timestamp:new Date().toISOString()};
  state.selectedObject=null;$('emptyState').hidden=true;$('previewCanvas').hidden=false;$('cameraPreview').hidden=true;
  $('maskControl').hidden=result.task!=='segment';$('previewInfo').textContent=`${result.width} × ${result.height} px · ${result.task==='segment'?'segmentation':'detection'} complete`;
  draw();renderResults();
}
async function runImage(){
  const generation=state.generation;state.busy=true;controls();
  try{const result=await api('/api/detect',{method:'POST',body:formFor(state.file)});await applyResult(result,generation);notify(`${result.detections.length} objects found. Select one to inspect it.`);}
  catch(error){notify(error.message,true);}finally{state.busy=false;controls();}
}
async function camera(){
  if(state.stream){stopCamera();$('cameraPreview').hidden=true;if(!state.result)$('emptyState').hidden=false;return;}
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('This browser cannot access a webcam. Open this page in Chrome or Edge on localhost.');
    state.stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
    $('cameraPreview').srcObject=state.stream;await $('cameraPreview').play();
    $('cameraPreview').hidden=false;$('previewCanvas').hidden=true;$('emptyState').hidden=true;$('cameraButton').textContent='Disable camera';$('previewInfo').textContent='Camera ready · click Start live detection';
  }catch(error){notify(error.name==='NotAllowedError'?'Camera permission was denied. Allow camera access in your browser to continue.':error.name==='NotFoundError'?'No camera was found on this device.':error.message,true);stopCamera();}
  controls();
}
async function startLive(){
  if(state.live){stopLive();return;}
  state.live=true;const generation=++state.generation;controls();
  const frame=document.createElement('canvas');
  try{
    while(state.live&&state.stream&&generation===state.generation){
      const video=$('cameraPreview');if(!video.videoWidth)throw new Error('Camera is not ready. Enable the camera again.');
      frame.width=video.videoWidth;frame.height=video.videoHeight;frame.getContext('2d').drawImage(video,0,0);
      const blob=await new Promise(resolve=>frame.toBlob(resolve,'image/jpeg',.85));
      const result=await api('/api/detect',{method:'POST',body:formFor(new File([blob],'camera-frame.jpg',{type:'image/jpeg'}))});
      await applyResult(result,generation);
      if(generation!==state.generation)break;
      await new Promise(resolve=>setTimeout(resolve,150));
    }
  }catch(error){notify(error.message,true);stopLive();}
  controls();
}
async function runVideo(){
  state.busy=true;controls();
  try{
    const response=await api('/api/videos',{method:'POST',body:formFor(state.file)});
    state.job=response.id;resetResults();$('jobPanel').hidden=false;$('cancelJob').hidden=false;
    $('jobBar').value=0;$('jobProgress').textContent='0%';$('jobLabel').textContent='Processing video';
    state.busy=false;controls();
    while(state.job){
      const job=await api(`/api/videos/${state.job}`);
      $('jobBar').value=job.progress;$('jobProgress').textContent=`${job.progress}%`;
      $('jobLabel').textContent=job.status==='encoding'?'Preparing your video':job.status==='complete'?'Analysis complete':'Processing video';
      $('jobDetail').textContent=job.frames?`${job.frames} / ${job.total_frames} frames · ${job.status}`:'Preparing the model…';
      if(job.status==='failed')throw new Error(job.error);
      if(job.status==='cancelled'){notify('Video analysis cancelled.');$('jobLabel').textContent='Analysis cancelled';break;}
      if(job.status==='complete'){
        state.result={...job,filename:state.file.name,task:$('taskSelect').value,timestamp:new Date().toISOString()};
        $('videoPreview').src=job.url;$('videoPreview').hidden=false;$('previewInfo').textContent='Annotated video · no audio';renderResults();inspectorTab('summary');notify('Video analysis complete. Play or download your result.');break;
      }
      await new Promise(resolve=>setTimeout(resolve,700));
    }
  }catch(error){notify(error.message,true);$('jobLabel').textContent='Video analysis interrupted';}
  finally{state.job=null;state.busy=false;$('cancelJob').hidden=true;controls();}
}
function download(blob,name){const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);}
function reportData(){const {image,...report}=state.result;return {...report,application:'YOLO Studio',count_semantics:report.detections?'objects in this frame':'detections summed across video frames'};}
async function refreshUsers(){
  const data=await api('/api/users');$('usersList').replaceChildren();
  for(const user of data.users){const row=document.createElement('div'),name=document.createElement('span');name.textContent=user;row.append(name);
    if(user!=='admin'){const button=document.createElement('button');button.className='quiet';button.textContent='Remove';button.onclick=async()=>{if(!confirm(`Remove user ${user}?`))return;try{await api('/api/users',jsonRequest('DELETE',{username:user}));await refreshUsers();}catch(error){$('accountError').textContent=error.message;}};row.append(button);} $('usersList').append(row);}
}
async function initializeUser(user){
  state.user=user;$('accountButton').textContent=user;$('engineStatus').textContent='Loading model';
  const meta=await api('/api/meta');state.classes=meta.classes;state.ready=true;renderClasses();$('engineStatus').textContent='Engine ready';controls();
}
$('loginDialog').addEventListener('cancel',event=>event.preventDefault());
$('loginForm').onsubmit=async event=>{event.preventDefault();$('loginSubmit').disabled=true;$('loginError').textContent='';try{const data=await api('/api/login',jsonRequest('POST',{username:$('username').value,password:$('password').value}));await initializeUser(data.user);$('password').value='';$('loginDialog').close();}catch(error){$('loginError').textContent=error.message;}finally{$('loginSubmit').disabled=false;}};
$('accountButton').onclick=async()=>{if(!state.user){$('loginDialog').showModal();return;}$('signedInAs').textContent=`Signed in as ${state.user}`;$('adminControls').hidden=state.user!=='admin';$('accountError').textContent='';$('accountDialog').showModal();if(state.user==='admin')try{await refreshUsers();}catch(error){$('accountError').textContent=error.message;}};
$('closeAccount').onclick=()=>$('accountDialog').close();
$('logoutButton').onclick=async()=>{if(state.job){$('accountError').textContent='Wait for the video or cancel it before signing out.';return;}try{stopCamera();await api('/api/logout',jsonRequest('POST',{}));state.user=null;state.ready=false;clearMedia();resetResults();$('accountDialog').close();$('loginDialog').showModal();controls();}catch(error){$('accountError').textContent=error.message;}};
$('addUserForm').onsubmit=async event=>{event.preventDefault();try{await api('/api/users',jsonRequest('POST',{username:$('newUsername').value,password:$('newPassword').value}));event.target.reset();await refreshUsers();$('accountError').textContent='';notify('User added.');}catch(error){$('accountError').textContent=error.message;}};
$('changePasswordForm').onsubmit=async event=>{event.preventDefault();try{await api('/api/users',jsonRequest('PUT',{current:$('currentPassword').value,password:$('changedPassword').value}));event.target.reset();$('accountError').textContent='';notify('Administrator password updated.');}catch(error){$('accountError').textContent=error.message;}};
document.querySelectorAll('[data-source]').forEach(button=>button.onclick=()=>changeSource(button.dataset.source));
document.querySelectorAll('[data-inspector]').forEach(button=>button.onclick=()=>inspectorTab(button.dataset.inspector));
$('uploadZone').onclick=()=>$('fileInput').click();$('fileInput').onchange=event=>acceptFile(event.target.files[0]);
['dragenter','dragover'].forEach(name=>$('uploadZone').addEventListener(name,event=>{event.preventDefault();$('uploadZone').classList.add('dragging');}));
['dragleave','drop'].forEach(name=>$('uploadZone').addEventListener(name,event=>{event.preventDefault();$('uploadZone').classList.remove('dragging');}));
$('uploadZone').addEventListener('drop',event=>acceptFile(event.dataTransfer.files[0]));
$('sampleButton').onclick=sample;$('emptySampleButton').onclick=sample;$('cameraButton').onclick=camera;
$('classSearch').oninput=renderClasses;$('clearClasses').onclick=()=>{state.selectedClasses.clear();$('classSearch').value='';renderClasses();markSettingsChanged();};
$('confidence').oninput=()=>{$('confidenceValue').value=`${$('confidence').value}%`;markSettingsChanged();};
$('taskSelect').onchange=()=>{$('taskHelp').textContent=$('taskSelect').value==='segment'?'Trace object shapes with pixel-level masks.':'Locate objects with precise bounding boxes.';markSettingsChanged();controls();};
['showBoxes','showLabels','showMasks'].forEach(id=>$(id).onchange=draw);
$('runButton').onclick=()=>{if($('runButton').disabled)return;if(state.source==='image')runImage();else if(state.source==='video')runVideo();else startLive();};
$('cancelJob').onclick=async()=>{if(!state.job)return;try{await api(`/api/videos/${state.job}`,{method:'DELETE'});$('jobLabel').textContent='Cancelling after the current frame…';}catch(error){notify(error.message,true);}};
$('previewCanvas').onclick=event=>{if(!state.result?.detections)return;const rect=event.target.getBoundingClientRect(),x=(event.clientX-rect.left)*event.target.width/rect.width,y=(event.clientY-rect.top)*event.target.height/rect.height;const hit=[...state.result.detections].reverse().find(d=>x>=d.box[0]&&x<=d.box[2]&&y>=d.box[1]&&y<=d.box[3]);selectObject(hit?hit.id:null);};
$('fitButton').onclick=()=>{state.selectedObject=null;draw();renderResults();if(document.fullscreenElement)document.exitFullscreen().catch(()=>{});};
$('fullscreenButton').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('previewStage').requestFullscreen();}catch{notify('Fullscreen is unavailable in this browser.',true);}};
$('themeButton').onclick=()=>{document.body.classList.toggle('light');try{localStorage.setItem('yolo-theme',document.body.classList.contains('light')?'light':'dark');}catch{}};
try{document.body.classList.toggle('light',localStorage.getItem('yolo-theme')==='light');}catch{}
$('downloadImage').onclick=()=>{if(state.result?.url){const link=document.createElement('a');link.href=state.result.url+'?download=1';link.click();}else $('previewCanvas').toBlob(blob=>download(blob,'yolo-studio-detection.png'));};
$('downloadJson').onclick=()=>download(new Blob([JSON.stringify(reportData(),null,2)],{type:'application/json'}),'yolo-studio-report.json');
$('downloadCsv').onclick=()=>{const r=state.result,rows=r.detections?[['object','confidence','x1','y1','x2','y2'],...r.detections.map(d=>[d.name,d.confidence,...d.box])]:[['object','frame_detections'],...Object.entries(r.counts)];download(new Blob([rows.map(row=>row.map(value=>`"${String(value).replaceAll('"','""')}"`).join(',')).join('\r\n')],{type:'text/csv'}),'yolo-studio-report.csv');};
$('printReport').onclick=()=>window.print();
document.addEventListener('keydown',event=>{
  if(event.ctrlKey&&event.key==='Enter'&&!$('loginDialog').open&&!$('accountDialog').open){event.preventDefault();$('runButton').click();return;}
  if(event.key==='Escape'&&!$('loginDialog').open&&!$('accountDialog').open){
    if(state.selectedObject!==null){selectObject(null);notify('Object highlight cleared.');}
    else if(state.live){stopLive();notify('Live detection stopped.');}
    else if(state.job){$('cancelJob').click();notify('Cancelling video analysis…');}
  }
});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&state.stream){stopCamera();notify('Camera stopped while the workspace is in the background.');}});
window.addEventListener('pagehide',()=>{state.stream?.getTracks().forEach(track=>track.stop());});
(async()=>{try{await api('/api/health');$('engineStatus').textContent='Engine online';const session=await api('/api/session');if(session.user)await initializeUser(session.user);else $('loginDialog').showModal();}catch(error){$('engineStatus').textContent='Connection issue';notify(error.message,true);if(!$('loginDialog').open)$('loginDialog').showModal();}})();
