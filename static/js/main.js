document.addEventListener('DOMContentLoaded', function() {
    loadVods();
    setupFormHandlers();
});

function loadVods() {
    const vodSelect = document.getElementById('vod_id');
    if (!vodSelect) return;
    
    fetch('/list_vods')
        .then(response => response.json())
        .then(vods => {
            vodSelect.innerHTML = '<option value="">Select a VOD</option>';
            vods.forEach(vodId => {
                const option = document.createElement('option');
                option.value = vodId;
                option.textContent = vodId;
                vodSelect.appendChild(option);
            });
        })
        .catch(error => console.error('Error loading VODs:', error));
}

function setupFormHandlers() {
    // Download form
    const downloadForm = document.getElementById('downloadForm');
    if (downloadForm) {
        downloadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(downloadForm);
            const resultsDiv = document.getElementById('results');
            const resultsContent = document.getElementById('resultsContent');
            
            resultsDiv.style.display = 'block';
            resultsContent.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"></div><p>Downloading chat...</p></div>';
            
            fetch('/download_twitch', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                let html = '<h4>Results:</h4><ul>';
                for (const [vodId, result] of Object.entries(data)) {
                    const status = result.success ? 'success' : 'danger';
                    html += `<li class="text-${status}">${vodId}: ${result.message}</li>`;
                }
                html += '</ul>';
                resultsContent.innerHTML = html;
                loadVods();
            })
            .catch(error => {
                resultsContent.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
        });
    }

    // Persona form
    const personaForm = document.getElementById('personaForm');
    if (personaForm) {
        personaForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(personaForm);
            const resultsDiv = document.getElementById('results');
            const resultsContent = document.getElementById('resultsContent');
            
            resultsDiv.style.display = 'block';
            resultsContent.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"></div><p>Generating personas...</p></div>';
            
            fetch('/generate_personas', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const vodId = formData.get('vod_id');
                    resultsContent.innerHTML = `
                        <div class="alert alert-success">Personas generated!</div>
                        <a href="/view_personas/${vodId}" class="btn btn-primary">View Personas</a>
                    `;
                } else {
                    resultsContent.innerHTML = `<div class="alert alert-danger">Error: ${data.message}</div>`;
                }
            })
            .catch(error => {
                resultsContent.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
        });
    }
}
