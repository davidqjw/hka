

// ====== State & Persistence ======
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const form = document.getElementById("composer");
const btn = document.getElementById("send");
const historyList = document.getElementById("historyList");
const newChatBtn = document.getElementById("newChat");

const SESS_KEY = 'hka.sessions';
const ACTIVE_KEY = 'hka.activeId';

//api path
const API_URL = '/api/chat';

/** Session shape: { id: string, title: string, created: number, messages: [{role, content}] } */
let sessions = loadSessions();
let activeId = localStorage.getItem(ACTIVE_KEY) || (sessions[0]?.id ?? null);

if (!activeId) {
    const s = createSession();
    sessions.push(s);
    activeId = s.id;
    persist();
}

function loadSessions() {
    try { return JSON.parse(localStorage.getItem(SESS_KEY)) || []; }
    catch { return []; }
}
function persist() {
    localStorage.setItem(SESS_KEY, JSON.stringify(sessions));
    localStorage.setItem(ACTIVE_KEY, activeId);
}
function createSession() {
    return { id: crypto.randomUUID(), title: 'New chat', created: Date.now(), messages: [] };
}
function getActive() { return sessions.find(s => s.id === activeId); }

// ====== UI Helpers ======
function addMessage(role, content, typing = false) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${role === "user" ? "user" : "bot"}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (typing) bubble.classList.add("typing");
    bubble.textContent = content;
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
    return bubble;
}

function clearChat() { chat.innerHTML = ''; }

function renderChatFromSession(session) {
    clearChat();
    for (const m of session.messages) {
        addMessage(m.role, m.content);
    }
}

// function renderSidebar() {
//     historyList.innerHTML = '';
//     // newest first
//     const sorted = [...sessions].sort((a, b) => b.created - a.created);
//     for (const s of sorted) {
//         const li = document.createElement('li');
//         li.className = 'sb-item' + (s.id === activeId ? ' active' : '');
//         li.dataset.id = s.id;
//         li.innerHTML = `<span class="dot"></span><span class="title" title="${s.title}">${s.title}</span>`;
//         li.addEventListener('click', () => {
//             activeId = s.id;
//             persist();
//             renderSidebar();
//             renderChatFromSession(getActive());
//         });
//         historyList.appendChild(li);
//     }
// }

function renderSidebar() {
    historyList.innerHTML = '';
    // newest first
    const sorted = [...sessions].sort((a, b) => b.created - a.created);

    for (const s of sorted) {
        const li = document.createElement('li');
        li.className = 'sb-item' + (s.id === activeId ? ' active' : '');
        li.dataset.id = s.id;

        li.innerHTML = `
          <span class="dot"></span>
          <span class="title" title="${s.title}">${s.title}</span>
          <button class="sb-del" title="Delete chat">✕</button>
        `;

        li.addEventListener('click', () => {
            activeId = s.id;
            persist();
            renderSidebar();
            renderChatFromSession(getActive());
        });

        // what: delete session
        const delBtn = li.querySelector('.sb-del');
        delBtn.addEventListener('click', (event) => {
            event.stopPropagation();

            sessions = sessions.filter(sess => sess.id !== s.id);

            if (activeId === s.id) {
                if (sessions.length > 0) {
                    const newest = [...sessions].sort((a, b) => b.created - a.created)[0];
                    activeId = newest.id;
                }
                else {
                    // what: new session if delete all
                    const newSess = createSession();
                    sessions.push(newSess);
                    activeId = newSess.id;
                }
            }

            persist();
            renderSidebar();
            renderChatFromSession(getActive());
        });

        historyList.appendChild(li);
    }
}

function ensureTitle(session) {
    if (session.title === 'New chat') {
        const firstUser = session.messages.find(m => m.role === 'user');
        if (firstUser) {
            session.title = (firstUser.content || 'New chat').slice(0, 40);
        }
    }
}

function setSending(sending) {
    btn.disabled = sending;
    input.disabled = sending;
}

// ====== App boot ======
renderSidebar();
renderChatFromSession(getActive());

// ====== Events ======
// Enter to send (Shift+Enter for newline)
input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        form.requestSubmit();
    }
});

newChatBtn.addEventListener('click', () => {
    const s = createSession();
    sessions.push(s);
    activeId = s.id;
    persist();
    renderSidebar();
    renderChatFromSession(s);
    input.focus();
});

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    const session = getActive();
    session.messages.push({ role: "user", content: text });
    addMessage("user", text);
    input.value = "";
    ensureTitle(session);
    persist();
    renderSidebar();

    const botBubble = addMessage("assistant", "…", true);
    setSending(true);

    try {
        const resp = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ messages: session.messages }),
        });

        if (!resp.ok) {
            const errorText = await resp.text();
            console.error("Server error:", errorText);
            botBubble.textContent = `Request failed: ${resp.status} ${resp.statusText}`;
            botBubble.classList.remove("typing");
            setSending(false);
            session.messages.push({ role: "assistant", content: botBubble.textContent });
            persist();
            return;
        }

        // json
        const data = await resp.json();
        let botText = "";
        // output_text is key
        botText = data.output_text; 

        botBubble.textContent = botText;
        botBubble.classList.remove("typing");
        chat.scrollTop = chat.scrollHeight;

        session.messages.push({ role: "assistant", content: botText });
        persist();
    }
    catch (err) {
        console.error("Fetch error:", err);
        botBubble.textContent = "Error: " + String(err.message || err);
        botBubble.classList.remove("typing");
        const session = getActive();
        session.messages.push({ role: "assistant", content: botBubble.textContent });
        persist();
    }
    finally {
        setSending(false);
    }
});