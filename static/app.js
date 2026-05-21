document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const queryInput = document.getElementById('query-input');
    const chatMessages = document.getElementById('chat-messages');
    const sendBtn = document.getElementById('send-btn');
    const historyList = document.getElementById('history-list');
    const currentSessionLabel = document.getElementById('current-session-id');
    const newChatBtn = document.getElementById('new-chat-btn');

    let currentSessionId = generateUUID();
    currentSessionLabel.textContent = `Session: ${currentSessionId.substring(0, 8)}`;

    loadSessions();

    newChatBtn.addEventListener('click', () => {
        currentSessionId = generateUUID();
        currentSessionLabel.textContent = `Session: ${currentSessionId.substring(0, 8)}`;
        chatMessages.innerHTML = `
            <div class="message assistant">
                <div class="message-content">
                    Hi, I'm the Deep Research Agent. Ask me a complex question, and I will search the web, compare sources, and synthesize a cited answer for you.
                </div>
            </div>
        `;
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // Add user message
        appendMessage('user', query);
        queryInput.value = '';
        sendBtn.disabled = true;

        // Create Assistant message container with Trace Inspector
        const assistantMsgDiv = document.createElement('div');
        assistantMsgDiv.className = 'message assistant';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Add Trace Inspector
        const template = document.getElementById('trace-inspector-template');
        const traceNode = template.content.cloneNode(true);
        const traceContainer = traceNode.querySelector('.trace-inspector');
        const traceTimeline = traceNode.querySelector('.trace-timeline');
        const ptSpan = traceNode.querySelector('.pt');
        const ctSpan = traceNode.querySelector('.ct');
        const sourcesGrid = traceNode.querySelector('.sources-grid');
        
        assistantMsgDiv.appendChild(traceContainer);
        assistantMsgDiv.appendChild(contentDiv);
        chatMessages.appendChild(assistantMsgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        let fullAnswer = "";

        try {
            const response = await fetch('/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query, session_id: currentSessionId })
            });

            if (!response.body) throw new Error('ReadableStream not supported');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop(); // Keep incomplete event in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6);
                        try {
                            const event = JSON.parse(dataStr);
                            handleEvent(event, contentDiv, traceTimeline, ptSpan, ctSpan, sourcesGrid);
                            
                            if (event.step === 'generating' && event.data) {
                                fullAnswer += event.data;
                                contentDiv.innerHTML = marked.parse(fullAnswer);
                            } else if (event.step === 'done') {
                                if (event.data && event.data.answer) {
                                    contentDiv.innerHTML = marked.parse(event.data.answer);
                                    ptSpan.textContent = event.data.prompt_tokens;
                                    ctSpan.textContent = event.data.completion_tokens;
                                    
                                    // Render sources
                                    if (event.data.doc_map) {
                                        sourcesGrid.innerHTML = '';
                                        for (const [doc_id, info] of Object.entries(event.data.doc_map)) {
                                            const [title, url, domain] = info;
                                            sourcesGrid.innerHTML += `
                                                <div class="source-card">
                                                    <a href="${url}" target="_blank">${title || domain}</a>
                                                    <div style="font-size:0.75rem; color: var(--text-muted)">${domain}</div>
                                                </div>
                                            `;
                                        }
                                    }
                                }
                                loadSessions(); // Refresh sidebar
                            } else if (event.step === 'error') {
                                contentDiv.innerHTML = `<span style="color:red">Error: ${event.message}</span>`;
                            }
                        } catch (e) {
                            console.error("Error parsing SSE:", e, dataStr);
                        }
                    }
                }
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        } catch (err) {
            contentDiv.innerHTML = `<span style="color:red">Connection failed: ${err.message}</span>`;
        } finally {
            sendBtn.disabled = false;
            // Remove spinners from timeline
            traceTimeline.querySelectorAll('.spinner').forEach(s => s.remove());
        }
    });

    function handleEvent(event, contentDiv, timeline, ptSpan, ctSpan, sourcesGrid) {
        if (event.step === 'done' || event.step === 'error' || event.step === 'generating') return;
        
        // Remove existing spinners
        timeline.querySelectorAll('.spinner').forEach(s => s.remove());
        
        const li = document.createElement('li');
        li.innerHTML = `<div class="spinner"></div> <span>${event.message}</span>`;
        timeline.appendChild(li);
        
        if (event.step === 'planning' && event.data && event.data.queries) {
            const qDiv = document.createElement('div');
            qDiv.style.fontSize = '0.75rem';
            qDiv.style.color = 'var(--text-muted)';
            qDiv.style.marginLeft = '22px';
            qDiv.textContent = `Queries: ${event.data.queries.join(', ')}`;
            timeline.appendChild(qDiv);
        }
    }

    function appendMessage(role, text) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        div.innerHTML = `<div class="message-content">${marked.parse(text)}</div>`;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function loadSessions() {
        try {
            const res = await fetch('/sessions');
            const sessions = await res.json();
            historyList.innerHTML = '';
            sessions.forEach(s => {
                const div = document.createElement('div');
                div.className = 'history-item';
                div.textContent = `${s.session_id.substring(0,8)} (${s.turn_count} turns)`;
                if (s.session_id === currentSessionId) div.classList.add('active');
                div.addEventListener('click', () => loadSessionHistory(s.session_id));
                historyList.appendChild(div);
            });
        } catch(e) {
            console.error("Failed to load sessions", e);
        }
    }

    async function loadSessionHistory(sessionId) {
        currentSessionId = sessionId;
        currentSessionLabel.textContent = `Session: ${currentSessionId.substring(0, 8)}`;
        try {
            const res = await fetch(`/sessions/${sessionId}/history`);
            const history = await res.json();
            
            chatMessages.innerHTML = '';
            history.forEach(turn => {
                appendMessage('user', turn.query);
                
                // Add Assistant msg with trace
                const assistantMsgDiv = document.createElement('div');
                assistantMsgDiv.className = 'message assistant';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.innerHTML = marked.parse(turn.response || 'No response');
                
                const template = document.getElementById('trace-inspector-template');
                const traceNode = template.content.cloneNode(true);
                const traceContainer = traceNode.querySelector('.trace-inspector');
                const timeline = traceNode.querySelector('.trace-timeline');
                const sourcesGrid = traceNode.querySelector('.sources-grid');
                
                // Reconstruct trace roughly
                if (turn.state_trace) {
                    try {
                        const states = JSON.parse(turn.state_trace);
                        states.forEach(state => {
                            const li = document.createElement('li');
                            li.textContent = `✓ ${state}`;
                            timeline.appendChild(li);
                        });
                    } catch(e) {}
                }
                
                if (turn.doc_map) {
                    try {
                        const docMap = JSON.parse(turn.doc_map);
                        for (const [doc_id, info] of Object.entries(docMap)) {
                            const [title, url, domain] = info;
                            sourcesGrid.innerHTML += `
                                <div class="source-card">
                                    <a href="${url}" target="_blank">${title || domain}</a>
                                    <div style="font-size:0.75rem; color: var(--text-muted)">${domain}</div>
                                </div>
                            `;
                        }
                    } catch(e) {}
                }
                
                assistantMsgDiv.appendChild(traceContainer);
                assistantMsgDiv.appendChild(contentDiv);
                chatMessages.appendChild(assistantMsgDiv);
            });
            chatMessages.scrollTop = chatMessages.scrollHeight;
            loadSessions();
        } catch (e) {
            console.error(e);
        }
    }

    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
});
