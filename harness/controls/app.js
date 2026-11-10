import * as THREE from '/three.js';

const stage=document.querySelector('#stage'), status=document.querySelector('#status');
const buttons=Object.fromEntries(['start','pause','reset','export'].map(k=>[k,document.getElementById(k)]));
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0x9eaeb1);
stage.append(renderer.domElement);renderer.domElement.tabIndex=0;
const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera();
scene.add(new THREE.HemisphereLight(0xffffff,0x64756a,2));
const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(10,25,-10);scene.add(light);
const objects=new Map(),keys=new Set();let mouse=[0,0],drag=false,running=false,busy=false,exportPending=false;
let state=await (await fetch('/state')).json(),catalog=await (await fetch('/catalog')).json();
if(state.job&&!state.job.running&&state.job.receipt?.accepted)state=await post('/reset');
let submitPending=false,jobRunning=Boolean(state.job?.running);
document.querySelector('#live').classList.toggle('enabled',state.world_bound);
const acceptedVideo=document.querySelector('#accepted-video');
function whiteboxView(){acceptedVideo.pause();acceptedVideo.style.display='none';renderer.domElement.style.display='block';draw();}
function videoView(){if(!state.accepted_video_available)return;acceptedVideo.src='/accepted-video?revision='+state.revision;renderer.domElement.style.display='none';acceptedVideo.style.display='block';acceptedVideo.play().catch(()=>{});}
document.querySelector('#view-whitebox').onclick=whiteboxView;
document.querySelector('#view-video').onclick=videoView;
document.querySelector('#view-video').disabled=!state.accepted_video_available;
function material(color){return new THREE.MeshStandardMaterial({color:new THREE.Color(...color),roughness:.9,side:THREE.DoubleSide});}
function asset(entity){
  const group=new THREE.Group(),data=catalog.assets[entity.asset_id || (entity.kind==='actor'?'gta_human':'')];
  if(data){
    for(const part of data.parts){
      const vertices=[];for(const face of catalog.faces){for(const i of [face[0],face[1],face[2],face[0],face[2],face[3]])vertices.push(...part.vertices[i]);}
      const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));geometry.computeVertexNormals();
      group.add(new THREE.Mesh(geometry,material(entity.color)));
    }
    group.userData.base=data.size;
  }else{
    group.add(new THREE.Mesh(new THREE.BoxGeometry(...entity.size),material(entity.color)));
  }
  scene.add(group);return group;
}
if(state.scene.static_geometry){
  const buffer=await (await fetch('/geometry')).arrayBuffer(),raw=new Float32Array(buffer);
  const interleaved=new THREE.InterleavedBuffer(raw,6),geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.InterleavedBufferAttribute(interleaved,3,0));
  geometry.setAttribute('color',new THREE.InterleavedBufferAttribute(interleaved,3,3));geometry.computeVertexNormals();
  scene.add(new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:1})));
}else{
  const t=state.scene.terrain,n=t.resolution,g=new THREE.PlaneGeometry(...t.size,n,n);g.rotateX(-Math.PI/2);
  const pos=g.attributes.position;
  for(let i=0;i<pos.count;i++){
    const x=pos.getX(i),z=pos.getZ(i),seed=t.seed;
    const h=t.height_amplitude*(.5*Math.sin(.31*x+seed*.013)*Math.cos(.27*z-seed*.017)+.25*Math.sin(.61*(x+z)+seed*.007));
    pos.setY(i,h*Math.max(0,Math.min(1,(Math.hypot(x,z)-(t.flatten_radius??5))/3)));
  }
  g.computeVertexNormals();scene.add(new THREE.Mesh(g,material([.42,.49,.42])));
}
function draw(){
  for(const entity of state.scene.entities){
    let obj=objects.get(entity.id);if(!obj){obj=asset(entity);objects.set(entity.id,obj);}
    obj.position.set(...entity.position);obj.rotation.set(0,entity.rotation_y||0,0);
    if(obj.userData.base){
      let height=1;if(entity.posture_weights)height=Object.entries(entity.posture_weights).reduce((v,[k,w])=>v+catalog.height_factors[k]*w,0);
      obj.scale.set(entity.size[0]/obj.userData.base[0],entity.size[1]*height/obj.userData.base[1],entity.size[2]/obj.userData.base[2]);
    }
    if(entity.kind==='staff'&&entity.weapon){
      const a=new THREE.Vector3(...entity.weapon.a),b=new THREE.Vector3(...entity.weapon.b),axis=b.clone().sub(a);
      obj.position.copy(a.add(b).multiplyScalar(.5));obj.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),axis.normalize());
    }
  }
  const c=state.scene.camera,k=c.intrinsics,ratio=k?k.width/k.height:16/9;
  const w=Math.min(stage.clientWidth,stage.clientHeight*ratio),h=w/ratio;
  renderer.setSize(w,h);camera.position.set(...c.position);camera.up.set(...(c.up||[0,1,0]));camera.lookAt(...c.target);
  camera.near=c.near||.05;camera.far=1000;camera.aspect=ratio;camera.fov=c.fov||60;camera.updateProjectionMatrix();
  if(k){
    const near=camera.near;camera.projectionMatrix.makePerspective(-k.cx/k.fx*near,(k.width-k.cx)/k.fx*near,k.cy/k.fy*near,-(k.height-k.cy)/k.fy*near,near,camera.far);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }
  renderer.render(scene,camera);
  document.querySelector('#clock').textContent=`${state.time.toFixed(3)} s · ${state.ticks} ticks`;
}
async function post(path,data={}){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const v=await r.json();if(!r.ok)throw Error(v.error);return v;}
function pause(){running=false;keys.clear();mouse=[0,0];buttons.start.disabled=false;buttons.pause.disabled=true;}
buttons.start.onclick=()=>{if(jobRunning)return;whiteboxView();running=true;buttons.start.disabled=true;buttons.pause.disabled=false;renderer.domElement.focus();status.textContent='预览中 · 未提交历史';};
buttons.pause.onclick=()=>{pause();status.textContent='已暂停';};
buttons.reset.onclick=async()=>{pause();exportPending=false;submitPending=false;try{state=await post('/reset');draw();status.textContent=state.world_bound?'已载入接受状态':'预览已重置';}catch(e){status.textContent=e.message;}};
buttons.export.onclick=()=>requestExport(false);
function lockJob(value){jobRunning=value;for(const id of ['start','reset','export','render-keys','render-prompt'])document.getElementById(id).disabled=value;}
async function submit(mode,goal=''){
  pause();const result=await post('/rollout',{mode,goal});lockJob(true);status.textContent='GPT / H3 处理中';status.title=result.job;
}
async function finishExport(){
  if(submitPending){await submit('keyboard');submitPending=false;}
  else{const result=await post('/export');pause();status.textContent=result.plan;}
  exportPending=false;
}
async function requestExport(generate){
  submitPending=generate;exportPending=true;
  if(!busy&&state.ticks>0&&(state.ticks+1)%17===5){
    try{await finishExport();}catch(e){exportPending=false;submitPending=false;status.textContent=e.message;}
  }else if(!running){
    exportPending=false;submitPending=false;status.textContent='记录尚未到提交边界，请继续录制';
  }else status.textContent='等待输入窗口结束';
}
document.getElementById('render-keys').onclick=()=>requestExport(true);
document.getElementById('render-prompt').onclick=async()=>{try{await submit('prompt',document.getElementById('action-prompt').value);}catch(e){status.textContent=e.message;}};
const mappings={KeyW:'w',KeyA:'a',KeyS:'s',KeyD:'d',ShiftLeft:'shift',ShiftRight:'shift',Space:'space',KeyC:'c',KeyX:'stand',KeyF:'attack'};
window.addEventListener('keydown',e=>{if(mappings[e.code]&&running){e.preventDefault();keys.add(mappings[e.code]);}});
window.addEventListener('keyup',e=>{if(mappings[e.code])keys.delete(mappings[e.code]);});
window.addEventListener('blur',()=>{pause();});
renderer.domElement.addEventListener('pointerdown',e=>{drag=true;renderer.domElement.setPointerCapture(e.pointerId);renderer.domElement.focus();});
renderer.domElement.addEventListener('pointerup',()=>drag=false);
renderer.domElement.addEventListener('pointermove',e=>{if(drag&&running){mouse[0]+=e.movementX;mouse[1]+=e.movementY;}});
window.addEventListener('resize',draw);
setInterval(async()=>{
  if(!running||busy)return;busy=true;const movement=mouse;mouse=[0,0];
  buttons.reset.disabled=true;
  try{
    state=await post('/step',{keys:[...keys],mouse:movement});draw();
    if(exportPending&&(state.ticks+1)%17===5){
      try{await finishExport();}
      catch(e){if(!/airborne|sweep|posture transition/.test(e.message))throw e;}
    }else if(!exportPending){const event=state.events.at(-1);status.textContent=event?`${event.type}${event.by?' · '+event.by:''}`:'预览中 · 未提交历史';}
  }catch(e){pause();status.textContent=e.message;}finally{busy=false;buttons.reset.disabled=jobRunning;}
},1000/24);
setInterval(async()=>{
  if(!jobRunning)return;
  try{
    const latest=await (await fetch('/state')).json();
    if(latest.job&&!latest.job.running){
      lockJob(false);
      if(latest.job.receipt?.accepted){state=await post('/reset');draw();document.querySelector('#view-video').disabled=false;videoView();status.textContent=`已接受并修正 · 世界版本 ${state.revision}`;}
      else {status.textContent='本段未接受';status.title=latest.job.directory;}
    }
  }catch(e){status.textContent=e.message;}
},2000);
draw();lockJob(jobRunning);status.textContent=jobRunning?'GPT / H3 处理中':
  (state.job&&!state.job.receipt?.accepted?'本段未接受':'就绪 · 未提交历史');
