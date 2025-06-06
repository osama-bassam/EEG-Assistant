  /* ============================================================================
      🧠 CHATBOT TOGGLE & POSITIONING
    ============================================================================ */
    const chatToggle = document.getElementById('chatToggle');
    const chatWrapper = document.getElementById('chatbot-wrapper');
    
    chatToggle.onclick = function () {
      const isVisible = chatWrapper.style.display === 'block';
    
      if (isVisible) {
        chatWrapper.style.display = 'none';
      } else {
        // Show and position next to icon
        chatWrapper.style.display = 'block';
        chatWrapper.style.position = 'fixed';
        chatWrapper.style.visibility = 'hidden';
    
        const rect = chatToggle.getBoundingClientRect();
        const width = chatWrapper.offsetWidth || 300;
        const height = chatWrapper.offsetHeight || 400;
    
        chatWrapper.style.left = `${rect.left - width}px`;
        chatWrapper.style.top = `${rect.top - height}px`;
        chatWrapper.style.right = 'auto';
        chatWrapper.style.bottom = 'auto';
        chatWrapper.style.visibility = 'visible';
      }
    
      // Stop pulsing after icon is clicked
      chatToggle.classList.remove('pulsing-intelligence');
    };
    
    // 🧠 Hide chatbot when clicking outside
    document.addEventListener('click', function (e) {
      if (!chatWrapper.contains(e.target) && !chatToggle.contains(e.target)) {
        chatWrapper.style.display = 'none';
      }
    });
    
    /* ============================================================================
      💬 CHATBOT ICON DRAGGABLE FUNCTIONALITY
    ============================================================================ */
    const icon = document.getElementById('chatToggle');
    let isDragging = false, offsetX, offsetY;
    
    icon.addEventListener('mousedown', e => {
      isDragging = true;
      offsetX = e.offsetX;
      offsetY = e.offsetY;
    });
    
    document.addEventListener('mousemove', e => {
      if (isDragging) {
        const x = e.pageX - offsetX;
        const y = e.pageY - offsetY;
        icon.style.left = `${x}px`;
        icon.style.top = `${y}px`;
        icon.style.right = 'auto';
        icon.style.bottom = 'auto';
        icon.style.position = 'fixed';
    
        // 🔁 Reposition chatbot wrapper relative to icon
        const wrapper = document.getElementById('chatbot-wrapper');
        const wrapperWidth = wrapper.offsetWidth || 300;
        const wrapperHeight = wrapper.offsetHeight || 400;
        wrapper.style.left = `${x - wrapperWidth}px`;
        wrapper.style.top = `${y - wrapperHeight}px`;
        wrapper.style.right = 'auto';
        wrapper.style.bottom = 'auto';
        wrapper.style.position = 'fixed';
      }
    });
    
    document.addEventListener('mouseup', () => isDragging = false);
    
/* ============================================================================
  🧠 CHATBOT MESSAGE SENDER (AJAX to Flask)
============================================================================ */
$(document).ready(function () {
  // ✅ Shared send function (used by both click & Enter key)
  function sendMessage() {
    const query = $('#userQuery').val().trim();
    if (!query) {
      alert("Please enter a query.");
      return;
    }

    const filehash = document.getElementById('filehash').value;

    // ⬇️ Add user message to chat
    const userMessage = `<p><strong>You:</strong> ${query}</p>`;
    $('#chatbox').append(userMessage);
    $('#chatbox').scrollTop($('#chatbox')[0].scrollHeight);
    $('#userQuery').val(''); // ✅ Clear input immediately

    // ⬇️ Send to Flask backend
    $.ajax({
      url: '/query',
      method: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({ query, filehash }),
      success: function (response) {
        const botMessage = `<p><strong>Bot:</strong> ${response.response}</p>`;
        $('#chatbox').append(botMessage);
        $('#chatbox').scrollTop($('#chatbox')[0].scrollHeight);
      },
      error: function () {
        const errorMessage = `<p><strong>Error:</strong> Something went wrong.</p>`;
        $('#chatbox').append(errorMessage);
        $('#chatbox').scrollTop($('#chatbox')[0].scrollHeight);
      }
    });
  }

  // 🔁 Allow Enter key to send
  document.getElementById('userQuery').addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendMessage(); // ✅ Now calls same function as button
    }
  });

  // 🖱️ Send on button click
  $('#sendQueryBtn').click(sendMessage);
});

    /* ============================================================================
      📈 POLLING + CHATBOT BUBBLE & PULSE ON COMPLETE
    ============================================================================ */
  function positionBubbleNearIcon(bubble) {
    document.body.appendChild(bubble); // must append to measure
    const iconRect = chatToggle.getBoundingClientRect();
    const bubbleWidth = bubble.offsetWidth;
    const bubbleHeight = bubble.offsetHeight;

    // Align right edge of bubble to left edge of icon
    bubble.style.left = `${iconRect.left - bubbleWidth - 10}px`; // 10px gap
    bubble.style.top = `${iconRect.top - bubbleHeight - 10}px`;  // 10px above
  }

    function pollProgress(filehash) {
      const bar = document.getElementById('barFill');
      const st = document.getElementById('statusText');
      const interval = setInterval(() => {
        fetch(`/progress?filehash=${filehash}`)
          .then(res => res.json())
          .then(data => {
            const { progress, complete } = data;
            let percent = Math.round(35 + (progress * 0.65));
            percent = Math.min(100, percent);
            bar.style.width = percent + '%';
            bar.textContent = percent + '%';
            st.textContent = `Processing: ${percent}%`;
    
            st.style.color = percent < 50 ? 'orange' : (percent < 90 ? 'teal' : 'limegreen');
    
            if (complete) {
              clearInterval(interval);
              st.textContent = 'Done!';
              document.getElementById('chatToggle').classList.add('pulsing-intelligence');
    
              // 💬 Auto bot message (typing... then suggestion)
              const bubble = document.createElement('div');
              bubble.style.position = 'fixed';
              bubble.style.zIndex = '1001';
              bubble.innerHTML = `
                <div style="
                  background: #007bff;
                  color: white;
                  padding: 10px 15px;
                  border-radius: 15px;
                  font-size: 14px;
                  max-width: 220px;
                  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                  position: relative;">
                  <span class="typing-dots">Typing<span>.</span><span>.</span><span>.</span></span>
                  <div style="
                    position: absolute;
                    bottom: -10px;
                    right: 20px;
                    width: 0;
                    height: 0;
                    border-left: 8px solid transparent;
                    border-right: 8px solid transparent;
                    border-top: 10px solid #007bff;">
                  </div>
                </div>
              `;
              positionBubbleNearIcon(bubble); // ✅ dynamically position it


    
              const spans = bubble.querySelectorAll("span");
              let dot = 0;
              const dotInterval = setInterval(() => {
                spans.forEach((s, i) => s.style.opacity = i === dot ? "1" : "0.2");
                dot = (dot + 1) % spans.length;
              }, 400);
    
              setTimeout(() => {
                clearInterval(dotInterval);
                const msg = bubble.querySelector('.typing-dots');
                bubble.innerHTML = `
                  <div style="
                    background: #007bff;
                    color: white;
                    padding: 10px 15px;
                    border-radius: 15px;
                    font-size: 14px;
                    max-width: 220px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                    position: relative;">
                    Want me to help you explain the result?
                    <div style="
                      position: absolute;
                      bottom: -10px;
                      right: 20px;
                      width: 0;
                      height: 0;
                      border-left: 8px solid transparent;
                      border-right: 8px solid transparent;
                      border-top: 10px solid #007bff;">
                    </div>
                  </div>`;
                positionBubbleNearIcon(bubble); // 🔁 Reposition bubble based on new height


              }, 4000);
    
              let remover = setTimeout(() => bubble.remove(), 14000);
              document.getElementById('chatToggle').addEventListener('click', () => {
                clearTimeout(remover);
                bubble.remove();
              });
    
              // 🎯 Load final results into UI
              fetch(`/result?filehash=${filehash}`)
                .then(res => res.text())
                .then(resultsHtml => {
                  document.getElementById('resultsContainer').innerHTML = resultsHtml;
    
                  const defaultSelected = [];
                  $('#channels option').each((i, el) => {
                    if (el.selected) defaultSelected.push(el.value);
                  });
                  $('#channels').val(defaultSelected);
    
                  $('#channels').multiselect({
                    includeSelectAllOption: true,
                    enableFiltering: true,
                    buttonWidth: '100%',
                    maxHeight: 200
                  });
    
                  document.getElementById('index').addEventListener('change', (e) => {
                    currentIndex = +e.target.value;
                    updatePlot();
                  });
    
                  $('#channels').on('change', updatePlot);
                  updatePlot();
                });
            }
          })
          .catch(err => {
            console.error('Progress polling error:', err);
          });
      }, 1000);
    }
    /* ============================================================================
      📊 EEG PLOT NAVIGATION FUNCTIONS (not chatbot-related)
    ============================================================================ */
    let currentIndex = 0;
    function navigate(step) {
      currentIndex = Math.max(0, currentIndex + step);
      updatePlot();
    }
    window.navigate = navigate;
    
    function updatePlot() {
      const hash = document.getElementById('filehash')?.value;
      if (!hash) return;
      const selected = $('#channels').val() || [];
      const url = `/plot?filehash=${hash}&index=${currentIndex}&channels=${selected.join(',')}`;
      document.getElementById('plotImg').src = url + `&t=${Date.now()}`;
      document.getElementById('index').value = currentIndex;
      document.getElementById('windowLabel').innerText = `Window ${currentIndex + 1}`;
    }
    
    /* ============================================================================
      📤 FILE UPLOAD & PROGRESS TRACKING (not chatbot-related)
    ============================================================================ */
    $(function () {
      document.getElementById('uploadForm').addEventListener('submit', function (e) {
        e.preventDefault();
        const file = document.getElementById('edfFile').files[0];
        if (!file) return alert('Please choose a file.');
    
        const fd = new FormData();
        fd.append('edfFile', file);
    
        const bar = document.getElementById('barFill');
        const st = document.getElementById('statusText');
        document.getElementById('progress').style.display = 'block';
        bar.style.width = '0%';
        st.textContent = 'Uploading EDF file...';
        st.style.color = 'orange';
    
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/', true);
    
        xhr.upload.onprogress = function (e) {
          if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 35);
            bar.style.width = percent + '%';
            bar.textContent = percent + '%';
          }
        };
    
        xhr.onload = function () {
          if (xhr.status === 200) {
            bar.style.width = '35%';
            st.textContent = 'Uploading to cloud storage & processing...';
            st.style.color = 'orange';
    
            const json = JSON.parse(xhr.responseText);
            const hash = json.filehash;
            pollProgress(hash);
          } else {
            st.textContent = 'Upload failed';
          }
        };
    
        xhr.onerror = function () {
          st.textContent = 'Upload error';
        };
    
        xhr.send(fd);
      });
    });
    
