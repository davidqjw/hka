 // ====== State & Persistence ======
    const chat   = document.getElementById("chat");
    const input  = document.getElementById("input");
    const form   = document.getElementById("composer");
    const btn    = document.getElementById("send");
    const historyList = document.getElementById("historyList");
    const newChatBtn  = document.getElementById("newChat");

    const SESS_KEY = 'hka.sessions';
    const ACTIVE_KEY = 'hka.activeId';

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
    function addMessage(role, content, typing=false) {
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

    // function timeAgo(ts) {
    //   const diff = Math.max(1, Math.floor((Date.now() - ts)/1000));
    //   const units = [
    //     ['yr', 31536000], ['mo', 2592000], ['d', 86400], ['h', 3600], ['m', 60], ['s', 1]
    //   ];
    //   for (const [name, sec] of units) {
    //     if (diff >= sec) return Math.floor(diff/sec) + name;
    //   }
    //   return 'now';
    // }

    function renderSidebar() {
      historyList.innerHTML = '';
      // newest first
      const sorted = [...sessions].sort((a,b) => b.created - a.created);
      for (const s of sorted) {
        const li = document.createElement('li');
        li.className = 'sb-item' + (s.id === activeId ? ' active' : '');
        li.dataset.id = s.id;
        li.innerHTML = `<span class="dot"></span><span class="title" title="${s.title}">${s.title}</span>`;
        li.addEventListener('click', () => {
          activeId = s.id;
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
        const resp = await fetch("http://localhost:3000/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: session.messages })
        });

        if (!resp.ok || !resp.body) {
          botBubble.textContent = "Request failed.";
          botBubble.classList.remove("typing");
          setSending(false);
          session.messages.push({ role: "assistant", content: botBubble.textContent });
          persist();
          return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let acc = "", botText = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          acc += decoder.decode(value, { stream: true });

          let idx;
          while ((idx = acc.indexOf("\n\n")) !== -1) {
            const event = acc.slice(0, idx).trim();
            acc = acc.slice(idx + 2);

            const dataLine = event.split("\n").find(l => l.startsWith("data:"));
            if (!dataLine) continue;

            const data = dataLine.slice(5).trim();
            if (data === "[DONE]") continue;

            try {
              const json = JSON.parse(data);
              if (typeof json.output_text === "string") {
                botText = json.output_text;          // full text
              } else if (json.delta && typeof json.delta === "string") {
                botText += json.delta;               // incremental
              }
              botBubble.textContent = botText;
              botBubble.classList.remove("typing");
              chat.scrollTop = chat.scrollHeight;
            } catch {}
          }
        }

        session.messages.push({ role: "assistant", content: botBubble.textContent });
        persist();
      } catch (err) {
        botBubble.textContent = "Error: " + String(err);
        botBubble.classList.remove("typing");
        const session = getActive();
        session.messages.push({ role: "assistant", content: botBubble.textContent });
        persist();
      } finally {
        setSending(false);
      }
    });