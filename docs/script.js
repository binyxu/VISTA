const blocks = [
  { id: "B03", type: "task", tokens: 9, state: "visible" },
  { id: "B17", type: "tool", tokens: 72, state: "visible" },
  { id: "B24", type: "file", tokens: 44, state: "archived" },
  { id: "B31", type: "note", tokens: 28, state: "visible" },
  { id: "B39", type: "web", tokens: 66, state: "visible" },
  { id: "B42", type: "trace", tokens: 55, state: "archived" },
];

const stack = document.getElementById("block-stack");
const fill = document.getElementById("pressure-fill");
const label = document.getElementById("pressure-label");

let tick = 0;

function render() {
  if (!stack || !fill || !label) return;
  stack.innerHTML = "";
  const pressure = 58 + ((tick * 7) % 31);
  fill.style.width = `${pressure}%`;
  label.textContent = `pressure ${pressure}%`;

  blocks.forEach((block, index) => {
    const dynamicTokens = Math.max(8, Math.min(92, block.tokens + (((tick + index) % 5) - 2) * 4));
    const archived = (tick + index) % 6 === 0 ? "archived" : block.state;
    const row = document.createElement("div");
    row.className = `context-block ${archived === "archived" ? "archived" : ""}`;
    row.innerHTML = `
      <span>${block.id} · ${block.type}</span>
      <span class="block-bar"><span style="width:${dynamicTokens}%"></span></span>
      <span>${archived}</span>
    `;
    stack.appendChild(row);
  });

  tick += 1;
}

render();
setInterval(render, 1800);
