
class FactCheckSelector {
    constructor() {
        this.isActive = false;
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.overlay = null;
        this.selectionBox = null;

        this.init();
    }

    init() {
        // Listen for messages from popup
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.action === 'toggleSelection') {
                this.toggleSelection();
                sendResponse({status: this.isActive ? 'active' : 'inactive'});
            }
        });

        // Create overlay and selection box elements
        this.createElements();

        console.log('FactCheck Selector content script loaded');
    }

    createElements() {
        // Create overlay
        this.overlay = document.createElement('div');
        this.overlay.id = 'factcheck-overlay';
        this.overlay.style.display = 'none';

        // Create selection box
        this.selectionBox = document.createElement('div');
        this.selectionBox.id = 'factcheck-selection-box';

        document.body.appendChild(this.overlay);
        document.body.appendChild(this.selectionBox);

        // Add event listeners
        this.overlay.addEventListener('mousedown', (e) => this.startSelection(e));
        document.addEventListener('mousemove', (e) => this.updateSelection(e));
        document.addEventListener('mouseup', (e) => this.endSelection(e));
        document.addEventListener('keydown', (e) => this.handleKeydown(e));
        this.selectionBox.addEventListener('dblclick', (e) => this.confirmSelection());
    }

    toggleSelection() {
        if (this.isActive) {
            this.deactivate();
        } else {
            this.activate();
        }
    }

    activate() {
        this.isActive = true;
        this.overlay.style.display = 'block';
        document.body.style.cursor = 'crosshair';
        console.log('Selection mode activated');
    }

    deactivate() {
        this.isActive = false;
        this.isDrawing = false;
        this.overlay.style.display = 'none';
        this.selectionBox.style.display = 'none';
        document.body.style.cursor = 'auto';
        console.log('Selection mode deactivated');
    }

    startSelection(e) {
        if (!this.isActive) return;

        this.isDrawing = true;
        this.startX = e.clientX + window.scrollX;
        this.startY = e.clientY + window.scrollY;

        this.selectionBox.style.left = this.startX + 'px';
        this.selectionBox.style.top = this.startY + 'px';
        this.selectionBox.style.width = '0px';
        this.selectionBox.style.height = '0px';
        this.selectionBox.style.display = 'block';

        e.preventDefault();
    }

    updateSelection(e) {
        if (!this.isDrawing) return;

        const currentX = e.clientX + window.scrollX;
        const currentY = e.clientY + window.scrollY;

        const width = Math.abs(currentX - this.startX);
        const height = Math.abs(currentY - this.startY);
        const left = Math.min(this.startX, currentX);
        const top = Math.min(this.startY, currentY);

        this.selectionBox.style.left = left + 'px';
        this.selectionBox.style.top = top + 'px';
        this.selectionBox.style.width = width + 'px';
        this.selectionBox.style.height = height + 'px';
    }

    endSelection(e) {
        if (!this.isDrawing) return;
        this.isDrawing = false;
        console.log('Selection ended - double click or press Enter to confirm');
    }

    handleKeydown(e) {
        if (!this.isActive) return;

        if (e.key === 'Escape') {
            this.deactivate();
            chrome.runtime.sendMessage({action: 'selectionCancelled'});
        } else if (e.key === 'Enter' && this.selectionBox.style.display === 'block') {
            this.confirmSelection();
        }
    }

    confirmSelection() {
        const rect = this.selectionBox.getBoundingClientRect();
        // Use viewport coordinates for bounding box
        const boundingBox = {
            x: rect.left,
            y: rect.top,
            width: rect.width,
            height: rect.height
        };

        // Extract content from the selected area
        const extractedData = this.extractContent(boundingBox);

        console.log('Extracted data:', extractedData);

        // Send to dummy API
        this.sendToAPI(extractedData);

        // Clean up
        this.deactivate();
        chrome.runtime.sendMessage({action: 'selectionConfirmed', data: extractedData});
    }

    extractContent(boundingBox) {
        // Get all elements within the bounding box (viewport coordinates)
        const allElements = document.querySelectorAll('p, span, div, img, video, a, h1, h2, h3, h4, h5, h6');
        const sel = boundingBox;
        const elementsInBounds = Array.from(allElements).filter(el => {
            const rect = el.getBoundingClientRect();

            // Skip bad tags
            const blacklist = ['HTML', 'BODY', 'HEAD', 'SCRIPT', 'STYLE', 'META', 'LINK'];
            if (blacklist.includes(el.tagName)) return false;

            // Skip if way too big (stricter)
            const selArea = sel.width * sel.height;
            const elArea = rect.width * rect.height;
            if (elArea > selArea * 2) return false;

            // Exclude elements that fully contain the selection (container divs, etc)
            if (
                rect.left <= sel.x &&
                rect.top <= sel.y &&
                rect.right >= sel.x + sel.width &&
                rect.bottom >= sel.y + sel.height
            ) {
                return false;
            }

            // Calculate intersection area
            const x_overlap = Math.max(0, Math.min(rect.right, sel.x + sel.width) - Math.max(rect.left, sel.x));
            const y_overlap = Math.max(0, Math.min(rect.bottom, sel.y + sel.height) - Math.max(rect.top, sel.y));
            const intersectionArea = x_overlap * y_overlap;
            // Only include if at least 50% of the element is inside the selection
            if (intersectionArea / elArea < 0.5) return false;

            return intersectionArea > 0;
        });

        // Extract different types of content
        const textParagraphs = [];
        const images = [];
        const videos = [];
        const links = [];
        let selectedHtml = '';

        elementsInBounds.forEach(el => {
            // Text content
            if (el.tagName === 'P' || el.tagName === 'DIV' || el.tagName === 'SPAN') {
                const text = el.innerText?.trim();
                if (text && text.length > 10) {
                    textParagraphs.push(text);
                }
            }

            // Images
            if (el.tagName === 'IMG') {
                images.push({
                    src: el.src,
                    alt: el.alt || '',
                    width: el.width,
                    height: el.height
                });
            }

            // Videos
            if (el.tagName === 'VIDEO') {
                videos.push({
                    src: el.src || el.currentSrc,
                    type: el.type || 'video',
                    duration: el.duration || 0
                });
            }

            // Links
            if (el.tagName === 'A') {
                links.push({
                    href: el.href,
                    text: el.innerText?.trim() || ''
                });
            }

            // Collect HTML
            selectedHtml += el.outerHTML || '';
        });

        return {
            timestamp: new Date().toISOString(),
            url: window.location.href,
            selectedHtml: selectedHtml,
            textParagraphs: [...new Set(textParagraphs)], // Remove duplicates
            images: images,
            videos: videos,
            links: links,
            boundingBox: boundingBox
        };
    }

    sendToAPI(data) {
        const dummyEndpoint = 'https://api.factcheck.example.com/analyze';

        console.log('Sending data to:', dummyEndpoint);
        console.log('Payload:', JSON.stringify(data, null, 2));

        fetch(dummyEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            console.log('API Response status:', response.status);
            return response.json();
        })
        .then(result => {
            console.log('API Response:', result);
        })
        .catch(error => {
            console.log('API call failed (expected for dummy endpoint):', error.message);
            console.log('Data would have been sent successfully to a real endpoint');
        });
    }
}

// Initialize the selector when the page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new FactCheckSelector());
} else {
    new FactCheckSelector();
}
