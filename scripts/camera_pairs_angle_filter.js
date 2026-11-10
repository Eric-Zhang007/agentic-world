const angleSelect=document.createElement('select');
angleSelect.id='angle';
angleSelect.setAttribute('aria-label','相机朝向差');
for(const [label,value] of [['全部角度','all'],['朝向差 ≤15°','15'],['朝向差 ≤30°','30'],['朝向差 ≤45°','45'],['朝向差 >45°','large']])angleSelect.add(new Option(label,value));
$('sample').before(angleSelect);
function filterWithAngle(){
  const value=angleSelect.value;
  list=D.samples.filter(x=>x.episode==$('episode').value&&(!$('kind').value||x.kind==$('kind').value)&&x.id.includes($('search').value)&&(value==='all'||(value==='large'?x.angle>45:x.angle<=Number(value))));
  $('sample').replaceChildren();
  for(const x of list)$('sample').add(new Option(`${x.kind} | ${x.age.toFixed(1)}s | ${x.angle.toFixed(1)}° | ${x.id}`,x.id));
  select();
}
angleSelect.onchange=filterWithAngle;
for(const id of ['episode','kind'])$(id).onchange=filterWithAngle;
$('search').oninput=filterWithAngle;
filterWithAngle();
