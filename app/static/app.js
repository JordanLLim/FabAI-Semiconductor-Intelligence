const $ = (x) => document.getElementById(x);

async function init() {
  const health = await fetch('/api/health').then((r) => r.json());
  $('status').textContent = health.dataset_ready ? 'WM-811K ONLINE' : 'DATASET REQUIRED';
  const summary = await fetch('/api/summary').then((r) => r.json());
  if (summary.ready) {
    $('wafers').textContent = summary.wafers.toLocaleString();
    $('lots').textContent = summary.lots.toLocaleString();
    run();
  }
}

async function run() {
  const response = await fetch('/api/incidents/Edge-Ring');
  if (!response.ok) {
    $('finding').textContent = 'Place LSWMD.pkl in data/raw/LSWMD.pkl to activate the real dataset.';
    return;
  }
  const data = await response.json();
  $('waferid').textContent = `${data.lot} / wafer ${data.wafer_index}`;
  $('pattern').textContent = data.label.toUpperCase();
  $('rate').textContent = (data.defect_rate * 100).toFixed(2) + '%';
  draw(data);

  const steps = [...document.querySelectorAll('#steps li')];
  steps.forEach((x) => x.classList.remove('active'));
  for (const step of steps) {
    await new Promise((resolve) => setTimeout(resolve, 350));
    step.classList.add('active');
  }

  $('finding').innerHTML = `<b>${data.label}</b> spatial signature identified. FAB.AI isolated the wafer and quantified its failing-die distribution. Equipment root-cause claims require separate process telemetry; this prototype does not invent telemetry absent from WM-811K.`;
}

function draw(data) {
  const canvas = $('wafer');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const sx = canvas.width / data.w;
  const sy = canvas.height / data.h;
  for (const [px, py, value] of data.cells) {
    ctx.fillStyle = value === 2 ? '#ff746c' : '#163b34';
    ctx.fillRect(px * sx, py * sy, Math.max(1, sx - 0.5), Math.max(1, sy - 0.5));
  }
}

$('investigate').onclick = run;
init();
