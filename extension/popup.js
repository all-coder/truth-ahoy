
document.addEventListener('DOMContentLoaded', function() {
    const toggleBtn = document.getElementById('toggleBtn');
    const status = document.getElementById('status');
    const lastExtraction = document.getElementById('lastExtraction');
    const extractionStats = document.getElementById('extractionStats');

    let isActive = false;

    // Update UI based on current state
    function updateUI(active) {
        isActive = active;

        if (active) {
            toggleBtn.textContent = 'Stop Selection';
            toggleBtn.className = 'btn btn--stop';
            status.textContent = 'Active - Select content on page';
            status.className = 'status status--active';
        } else {
            toggleBtn.textContent = 'Start Selection';
            toggleBtn.className = 'btn btn--primary';
            status.textContent = 'Inactive';
            status.className = 'status status--inactive';
        }
    }

    // Toggle selection mode
    toggleBtn.addEventListener('click', function() {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            chrome.tabs.sendMessage(tabs[0].id, {action: 'toggleSelection'}, function(response) {
                if (chrome.runtime.lastError) {
                    console.error('Error:', chrome.runtime.lastError.message);
                    status.textContent = 'Error - Refresh page and try again';
                    status.className = 'status status--inactive';
                    return;
                }

                updateUI(response.status === 'active');
            });
        });
    });

    // Listen for messages from content script
    chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
        if (request.action === 'selectionConfirmed') {
            updateUI(false);
            showLastExtraction(request.data);
        } else if (request.action === 'selectionCancelled') {
            updateUI(false);
        }
    });

    function showLastExtraction(data) {
        const stats = [
            `${data.textParagraphs?.length || 0} paragraphs`,
            `${data.images?.length || 0} images`, 
            `${data.videos?.length || 0} videos`,
            `${data.links?.length || 0} links`
        ].join(', ');

        extractionStats.textContent = stats;
        lastExtraction.style.display = 'block';

        console.log('Extraction completed:', data);
    }

    // Initialize UI
    updateUI(false);
});
