const chatForm = document.getElementById("chatForm");
const queryInput = document.getElementById("queryInput");
const messages = document.getElementById("messages");
const sources = document.getElementById("sources");
const sendButton = document.getElementById("sendButton");
const clearButton = document.getElementById("clearButton");
const healthText = document.getElementById("healthText");
const messageTemplate = document.getElementById("messageTemplate");
const sourceTemplate = document.getElementById("sourceTemplate");

const controls = {
  top_k: document.getElementById("topK"),
  vector_candidates: document.getElementById("vectorCandidates"),
  keyword_candidates: document.getElementById("keywordCandidates"),
  rerank_top_n: document.getElementById("rerankTopN"),
};

document.querySelectorAll(".prompt-chip").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.prompt ?? "";
    queryInput.focus();
  });
});

clearButton.addEventListener("click", () => {
  messages.innerHTML = "";
  sources.innerHTML = '<div class="empty-state">Supporting passages will appear here with section metadata, hybrid scores, and rerank scores after the first retrieval.</div>';
  appendMessage(
    "assistant",
    "I’m connected to the scientific retrieval pipeline. Ask about methods, cohorts, biomarkers, limitations, or reported findings, and I’ll answer from reranked evidence with chunk citations."
  );
});

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }

  appendMessage("user", query);
  queryInput.value = "";
  setBusy(true);
  const typingNode = appendTypingIndicator();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query,
        top_k: numberValue(controls.top_k.value),
        vector_candidates: numberValue(controls.vector_candidates.value),
        keyword_candidates: numberValue(controls.keyword_candidates.value),
        rerank_top_n: numberValue(controls.rerank_top_n.value),
      }),
    });

    const payload = await response.json();
    typingNode.remove();

    if (!response.ok) {
      appendMessage("assistant", payload.error || "The server could not answer that query.");
      renderSources([]);
      return;
    }

    appendMessage("assistant", payload.answer || "No answer returned.");
    renderSources(payload.retrieved_chunks || []);
  } catch (error) {
    typingNode.remove();
    appendMessage(
      "assistant",
      "The local UI could not reach the RAG backend. Make sure the Python server is running and the index is available."
    );
    renderSources([]);
  } finally {
    setBusy(false);
  }
});

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const payload = await response.json();
    healthText.textContent = payload.pipeline_ready
      ? "Ready for local requests"
      : "Pipeline not initialized";
  } catch (error) {
    healthText.textContent = "Server unavailable";
  }
}

function appendMessage(role, content) {
  const fragment = messageTemplate.content.cloneNode(true);
  const node = fragment.querySelector(".message");
  const avatar = fragment.querySelector(".avatar");
  const bubble = fragment.querySelector(".bubble");

  node.classList.add(role);
  avatar.textContent = role === "user" ? "YOU" : "AI";
  bubble.textContent = content;

  messages.appendChild(fragment);
  messages.scrollTop = messages.scrollHeight;
}

function appendTypingIndicator() {
  const article = document.createElement("article");
  article.className = "message assistant typing reveal";
  article.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>
  `;
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function renderSources(items) {
  if (!Array.isArray(items) || items.length === 0) {
    sources.innerHTML = '<div class="empty-state">No supporting chunks were returned for this response.</div>';
    return;
  }

  sources.innerHTML = "";
  items.forEach((item, index) => {
    const fragment = sourceTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".source-card");
    const meta = fragment.querySelector(".source-meta");
    const text = fragment.querySelector(".source-text");
    const hybridScore = formatScore(item.hybrid_score);
    const rerankScore = item.rerank_score == null ? "n/a" : item.rerank_score;

    card.style.animationDelay = `${index * 70}ms`;
    meta.innerHTML = `
      <strong>${escapeHtml(item.chunk_id || "Unknown chunk")}</strong><br />
      ${escapeHtml(item.document_id || "Unknown document")} • ${escapeHtml(item.section_path || "Unknown section")}<br />
      hybrid ${hybridScore} • rerank ${escapeHtml(String(rerankScore))}
    `;
    text.textContent = item.text || "";
    sources.appendChild(fragment);
  });
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  sendButton.textContent = isBusy ? "Retrieving..." : "Run Retrieval";
}

function numberValue(raw) {
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function formatScore(value) {
  if (typeof value !== "number") {
    return "n/a";
  }
  return value.toFixed(4);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

appendMessage(
  "assistant",
  "I’m connected to the scientific retrieval pipeline. Ask about methods, cohorts, biomarkers, limitations, or reported findings, and I’ll answer from reranked evidence with chunk citations."
);
checkHealth();
