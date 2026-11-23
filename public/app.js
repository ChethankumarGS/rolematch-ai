const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const savedPanel = document.getElementById("savedPanel");
const savedList = document.getElementById("savedList");
const savedCount = document.getElementById("savedCount");

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollDown() {
  chat.scrollTop = chat.scrollHeight;
}

function addMessage(role, text) {
  const msg = el("div", `msg ${role}`);
  const bubble = el("div", "bubble");
  if (role === "assistant") {
    // minimal safe markdown: escape everything, then restore **bold** markers
    const esc = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    bubble.innerHTML = esc.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  } else {
    bubble.textContent = text;
  }
  msg.appendChild(bubble);
  chat.appendChild(msg);
  scrollDown();
  return msg;
}

function fmtSalary(job) {
  if (job.salary_min_lpa == null && job.salary_max_lpa == null) return null;
  const fmt = (n) => (n == null ? "?" : Number(n.toFixed(1)));
  return `₹${fmt(job.salary_min_lpa)}–${fmt(job.salary_max_lpa)} LPA`;
}

function jobCard(job) {
  const card = el("div", "card");

  const top = el("div", "card-top");
  const titleWrap = el("div");
  titleWrap.appendChild(el("h3", null, job.title));
  titleWrap.appendChild(el("div", "company", job.company));
  top.appendChild(titleWrap);
  if (job.match_score != null) {
    top.appendChild(el("div", "score", `${Math.round(job.match_score * 100)}% match`));
  }
  card.appendChild(top);

  const meta = el("div", "meta");
  const bits = [
    job.location,
    job.work_mode,
    fmtSalary(job),
    job.experience_max != null ? `${job.experience_min}–${job.experience_max} yrs` : null,
    job.posted_days_ago != null
      ? job.posted_days_ago === 0 ? "posted today" : `posted ${job.posted_days_ago}d ago`
      : null,
  ].filter(Boolean);
  for (const bit of bits) meta.appendChild(el("span", null, bit));
  card.appendChild(meta);

  if (job.match_reasons && job.match_reasons.length) {
    const reasons = el("div", "reasons");
    reasons.appendChild(el("strong", null, "Why it fits: "));
    reasons.appendChild(document.createTextNode(job.match_reasons.join(" · ")));
    card.appendChild(reasons);
  }

  if (job.skills && job.skills.length) {
    const chips = el("div", "skillchips");
    for (const s of job.skills.slice(0, 8)) chips.appendChild(el("span", null, s));
    card.appendChild(chips);
  }

  const saveBtn = el("button", "save", "Save");
  saveBtn.addEventListener("click", async () => {
    if (saveBtn.classList.contains("saved")) {
      await fetch(`/api/save/${job.job_id}`, { method: "DELETE" });
      saveBtn.classList.remove("saved");
      saveBtn.textContent = "Save";
    } else {
      await fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: job.job_id }),
      });
      saveBtn.classList.add("saved");
      saveBtn.textContent = "Saved";
    }
    refreshSavedCount();
  });
  card.appendChild(saveBtn);

  return card;
}

function addJobCards(jobs) {
  const wrap = el("div", "msg assistant");
  const cards = el("div", "cards");
  const label = el("div", "demo-label", "Sample listings - demo data, not live openings");
  cards.appendChild(label);
  for (const job of jobs) cards.appendChild(jobCard(job));
  wrap.appendChild(cards);
  chat.appendChild(wrap);
  scrollDown();
}

async function refreshJobCount() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    if (d && typeof d.jobs === "number") {
      document.getElementById("jobCount").textContent = d.jobs.toLocaleString();
    }
  } catch (e) { /* leave placeholder */ }
}
refreshJobCount();

function addTyping() {
  const msg = el("div", "msg assistant");
  const bubble = el("div", "bubble typing");
  bubble.appendChild(el("span"));
  bubble.appendChild(el("span"));
  bubble.appendChild(el("span"));
  msg.appendChild(bubble);
  chat.appendChild(msg);
  scrollDown();
  return msg;
}

async function sendMessage(text) {
  addMessage("user", text);
  const typing = addTyping();
  sendBtn.disabled = true;
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await resp.json();
    typing.remove();
    if (data.reply) addMessage("assistant", data.reply);
    if (data.jobs && data.jobs.length) addJobCards(data.jobs);
  } catch (err) {
    typing.remove();
    addMessage("assistant", "Something went wrong reaching the server - try again.");
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  sendMessage(text);
});

document.getElementById("chips").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-prompt]");
  if (btn) sendMessage(btn.dataset.prompt);
});

async function refreshSavedCount() {
  try {
    const resp = await fetch("/api/saved");
    const data = await resp.json();
    savedCount.textContent = data.count;
    return data.jobs;
  } catch {
    return [];
  }
}

document.getElementById("savedToggle").addEventListener("click", async () => {
  savedPanel.hidden = false;
  const jobs = await refreshSavedCount();
  savedList.innerHTML = "";
  if (!jobs || !jobs.length) {
    savedList.appendChild(el("p", "empty", "Nothing saved yet. Ask me to save a role you like."));
  } else {
    for (const job of jobs) savedList.appendChild(jobCard(job));
  }
});

document.getElementById("savedClose").addEventListener("click", () => {
  savedPanel.hidden = true;
});

refreshSavedCount();
