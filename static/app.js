let currentTask = null;

async function loadNextTask() {
    showState('loading-state');
    
    try {
        const response = await fetch('/api/task');
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
    document.getElementById('baseline-lchair').checked = false;
    document.getElementById('mop-lchair').checked = false;
    document.getElementById('annotation-notes').value = '';
    
    // Set content
    document.getElementById('task-image').src = `/images/${data.image_filename}`;
    document.getElementById('baseline-text').textContent = data.baseline_caption;
    document.getElementById('mop-text').textContent = data.mop_caption;
}

async function submitAnnotation(winner) {
    if (!currentTask) return;
    
    const annotationData = {
        id: currentTask.id,
        winner: winner,
        baseline_lchair: document.getElementById('baseline-lchair').checked,
        mop_lchair: document.getElementById('mop-lchair').checked,
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
    try {
        const response = await fetch('/api/stats');
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
                submitAnnotation('baseline');
                break;
            case '2':
                submitAnnotation('mop');
                break;
            case '3':
                submitAnnotation('tie');
                break;
        }
    }
});

// Initial load
document.addEventListener('DOMContentLoaded', loadNextTask);
