let currentTask = null;
let currentUser = null;

function selectUser(userName) {
    currentUser = userName;
    document.getElementById('user-modal').style.display = 'none';
    document.getElementById('current-user-badge').textContent = userName;
    loadNextTask();
}

async function loadNextTask() {
    if (!currentUser) return;
    showState('loading-state');
    
    try {
        const response = await fetch(`/api/task?user=${encodeURIComponent(currentUser)}`);
        const data = await response.json();
        
        if (data.status === 'complete') {
            showState('complete-state');
            updateProgress();
            return;
        }
        
        currentTask = data;
        renderTask(data);
        updateProgress();
        showState('annotation-task');
        
    } catch (error) {
        console.error('Failed to load task:', error);
        alert('Failed to load next task. Please check server connection.');
    }
}

function renderTask(data) {
    // Reset inputs
    document.getElementById('annotation-notes').value = '';
    
    // Set content
    document.getElementById('task-image').src = `/images/${data.image_filename}`;
    
    // Randomize for blinded evaluation
    const isBaselineA = Math.random() < 0.5;
    currentTask.isBaselineA = isBaselineA;
    
    if (isBaselineA) {
        document.getElementById('model-a-text').innerHTML = `${data.baseline_caption_kr ? data.baseline_caption_kr.replace(/\\n/g, '<br>') : ''}<hr style="margin:10px 0; border:0; border-top:1px solid #eee;"><small style="color: #666;">${data.baseline_caption.replace(/\\n/g, '<br>')}</small>`;
        document.getElementById('model-b-text').innerHTML = `${data.mop_caption_kr ? data.mop_caption_kr.replace(/\\n/g, '<br>') : ''}<hr style="margin:10px 0; border:0; border-top:1px solid #eee;"><small style="color: #666;">${data.mop_caption.replace(/\\n/g, '<br>')}</small>`;
    } else {
        document.getElementById('model-a-text').innerHTML = `${data.mop_caption_kr ? data.mop_caption_kr.replace(/\\n/g, '<br>') : ''}<hr style="margin:10px 0; border:0; border-top:1px solid #eee;"><small style="color: #666;">${data.mop_caption.replace(/\\n/g, '<br>')}</small>`;
        document.getElementById('model-b-text').innerHTML = `${data.baseline_caption_kr ? data.baseline_caption_kr.replace(/\\n/g, '<br>') : ''}<hr style="margin:10px 0; border:0; border-top:1px solid #eee;"><small style="color: #666;">${data.baseline_caption.replace(/\\n/g, '<br>')}</small>`;
    }
}

async function submitAnnotation(winner_btn) {
    if (!currentTask) return;
    
    let actual_winner = 'tie';
    if (winner_btn === 'A') {
        actual_winner = currentTask.isBaselineA ? 'baseline' : 'mop';
    } else if (winner_btn === 'B') {
        actual_winner = currentTask.isBaselineA ? 'mop' : 'baseline';
    }
    
    const annotationData = {
        id: currentTask.id,
        user: currentUser,
        winner: actual_winner,
        baseline_lchair: false,
        mop_lchair: false,
        notes: document.getElementById('annotation-notes').value
    };
    
    try {
        const response = await fetch('/api/annotate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(annotationData)
        });
        
        if (response.ok) {
            loadNextTask();
        } else {
            throw new Error('Server returned error');
        }
    } catch (error) {
        console.error('Failed to submit annotation:', error);
        alert('Failed to save annotation. Please try again.');
    }
}

async function updateProgress() {
    if (!currentUser) return;
    try {
        const response = await fetch(`/api/stats?user=${encodeURIComponent(currentUser)}`);
        const data = await response.json();
        
        const text = `Progress: ${data.completed} / ${data.total}`;
        document.getElementById('progress-text').textContent = text;
        
        const percentage = data.total > 0 ? (data.completed / data.total) * 100 : 0;
        document.getElementById('progress-fill').style.width = `${percentage}%`;
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

function showState(stateId) {
    document.querySelectorAll('.main-content > div').forEach(el => {
        el.classList.remove('active');
    });
    document.getElementById(stateId).classList.add('active');
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Only if not typing in text area
    if (e.target.tagName.toLowerCase() === 'textarea') return;
    
    if (document.getElementById('annotation-task').classList.contains('active')) {
        switch(e.key) {
            case '1':
                submitAnnotation('A');
                break;
            case '2':
                submitAnnotation('B');
                break;
            case '3':
                submitAnnotation('tie');
                break;
        }
    }
});

// Initial load
document.addEventListener('DOMContentLoaded', () => {
    // Wait for user to select identity from modal
});
