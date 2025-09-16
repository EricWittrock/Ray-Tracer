document.body.style.backgroundColor = 'black';
let canvas = document.createElement('canvas');
canvas.width = 800;
canvas.height = 800;
document.body.appendChild(canvas);
let ctx = canvas.getContext('2d');

const N = 25;

function Node() {
    this.h = Math.random();
    this.bh = Math.random();
    this.adj = [];
    this.x = Math.random() * canvas.width;
    this.y = Math.random() * canvas.height;
    this.vx = 0;
    this.vy = 0;
}

let nodes = [];
for(let i = 0; i<N; i++) {
    let newNode = new Node();
    nodes.push(newNode);
}

for(let i = 0; i<N; i++) {
    for(let j = 0; j<i; j++) {
        if(Math.random() < 0.3) {
            let w = Math.random();
            nodes[i].adj.push([nodes[j], w]);
            nodes[j].adj.push([nodes[i], w]);
        }
    }
}

function step() {
    const alpha = 0.5;
    for(let i = 0; i<N; i++) {
        let node = nodes[i];
        let num_adj = node.adj.length;
        let sum_w = 0;
        for(let a = 0; a < num_adj; a++) {
            sum_w += nodes[i].adj[a][1];
        }
        let wh_sum = 0;
        for(let a = 0; a < num_adj; a++) {
            let adj = nodes[i].adj[a][0];
            let w = nodes[i].adj[a][1];
            let adj_h = adj.h;
            wh_sum += w * adj_h;
        }
        node.h = alpha * node.h + (1-alpha) * (wh_sum / sum_w);
        node.h = alpha * node.bh + (1-alpha) * node.h
    }
}

function steps(num_steps=1000) {
    for(let i = 0; i<num_steps; i++) {
        step();
    }
}

function grad(node) {
    let k = 0.00001;
    let grad = new Array(N).fill(0);
    let original_h = nodes.map(i=>i.h);
    let h0 = node.h;
    for(let i = 0; i<N; i++) {
        nodes[i].bh += k;
        steps();
        let h1 = node.h;
        let g = (h1-h0) / k;
        grad[i] = g;
        console.log(g);
        for(let i = 0; i<N; i++) {
            nodes[i].h = original_h[i];
        }
        nodes[i].bh -= k;
    }
    return grad;
}

function drawNode(color, node) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, 15, 0, 2 * Math.PI);
    ctx.fill();
}

function colorMap(t) {
  // Clamp t to [0,1]
  t = Math.max(0, Math.min(1, t));

  // Define stops in HSL:
  // white: h=0, s=0%, l=100%
  // soft yellow: h=50°, s=100%, l=70%
  // dark purple: h=280°, s=70%, l=20%
  const stops = [
    { t: 0.0, h:   0, s: 0,   l: 100 }, // white
    { t: 0.4, h:  50, s: 100, l: 70  }, // soft yellow
    { t: 1.0, h: 280, s: 70,  l: 20  }  // dark purple
  ];

  // Find which two stops t falls between
  let i = 0;
  while (i < stops.length - 1 && t > stops[i + 1].t) {
    i++;
  }

  const s0 = stops[i];
  const s1 = stops[i + 1];
  const localT = (t - s0.t) / (s1.t - s0.t);

  // Interpolate hue carefully (circular)
  let hDiff = s1.h - s0.h;
  if (Math.abs(hDiff) > 180) {
    hDiff -= Math.sign(hDiff) * 360;
  }
  const h = (s0.h + hDiff * localT + 360) % 360;

  const s = s0.s + (s1.s - s0.s) * localT;
  const l = s0.l + (s1.l - s0.l) * localT;

  return `hsl(${h.toFixed(1)}, ${s.toFixed(1)}%, ${l.toFixed(1)}%)`;
}



function drawNodes() {
    for(let i = 0; i<N; i++) {
        let node = nodes[i];
        let color = colorMap(node.h);
        drawNode(color, node);
    }
}

function drawEdges() {
    
    for(let i = 0; i<N; i++) {
        let node = nodes[i];
        for(let a = 0; a<node.adj.length; a++) {
            // stroke style depending on weight
            let w = node.adj[a][1];
            ctx.strokeStyle = `rgba(255, 255, 255, ${w})`;
            ctx.lineWidth = 2 * w;
            let adj = node.adj[a][0];
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(adj.x, adj.y);
            ctx.stroke();
        }
    }
}

function nodesStepPos() {
    // move each node so nodes with less weight have shorter edges
    for(let i = 0; i<N; i++) {
        let node = nodes[i];
        let fx = 0;
        let fy = 0;
        for(let a = 0; a<node.adj.length; a++) {
            let adj = node.adj[a][0];
            let w = node.adj[a][1];
            let dx = adj.x - node.x;
            let dy = adj.y - node.y;
            let dist = Math.sqrt(dx*dx + dy*dy) * 0.5;
            let desired_dist = 200 * (1 - w);
            let diff = dist - desired_dist;
            if(dist > 0) {
                fx += (diff * dx / dist);
                fy += (diff * dy / dist);
            }

            // repulsion
            let repulsion = 10000;
            let r = Math.sqrt(dx*dx + dy*dy);
            if(r > 0) {
                fx -= (repulsion * dx / (r*r));
                fy -= (repulsion * dy / (r*r));
            }
        }
        fx *= 0.01;
        fy *= 0.01;
        
        node.vx += fx;
        node.vy += fy;
        // friction
        node.vx *= 0.85;
        node.vy *= 0.85;
        // update position
        node.x += node.vx;
        node.y += node.vy;
        // keep in bounds
        node.x = Math.max(20, Math.min(canvas.width - 20, node.x));
        node.y = Math.max(20, Math.min(canvas.height - 20, node.y));
    }
}

function randomizeH() {
    for(let i = 0; i<N; i++) {
        nodes[i].h = Math.random();
    }
}

steps(1000);
let hs = nodes.map(i=>i.h);
randomizeH();
let str = '';
for(let i = 0; i<=20; i++) {
    let nhs = nodes.map(i=>i.h);
    let sum_diff = 0;
    for(let j = 0; j<N; j++) {
        sum_diff += Math.abs(hs[j] - nhs[j]);
    }
    str += i + ", " + sum_diff + "\n";
    steps(1);
}
console.log(str);



// setInterval(()=>{
//     loop();
// }, 100);

function loop(s) {
    for(let i = 0; i<s; i++) {
        steps(1);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawEdges();
        drawNodes();
        nodesStepPos();
    }
}