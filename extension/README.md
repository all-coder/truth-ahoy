# FactCheck Selector Browser Extension

A barebones Chrome extension that allows users to select content on web pages using a bounding box and extract it for fact-checking analysis.

## Features

- Draw bounding boxes on any webpage
- Extract content from selected areas including:
  - Text paragraphs
  - Images (with URLs and alt text)
  - Videos (with URLs and metadata)
  - Links (with URLs and text)
  - Raw HTML content
- Send extracted data as JSON to a configurable API endpoint
- Simple, lightweight implementation with no complex styling

## Installation

1. Download or clone this extension
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" in the top right
4. Click "Load unpacked" and select the extension folder
5. The extension icon should appear in your toolbar

## How to Use

1. Click the extension icon in your browser toolbar
2. Click "Start Selection" in the popup
3. The page will show a semi-transparent overlay
4. Click and drag to create a bounding box around the content you want to extract
5. Double-click the selection or press Enter to confirm
6. The extension will extract the content and send it to the API endpoint
7. Press Escape to cancel selection at any time

## API Integration

The extension sends extracted data as JSON to: `https://api.factcheck.example.com/analyze`

### JSON Structure
```json
{
  "timestamp": "2025-08-30T12:00:00.000Z",
  "url": "https://example.com/page",
  "selectedHtml": "<div>raw HTML content</div>",
  "textParagraphs": ["paragraph 1", "paragraph 2"],
  "images": [
    {
      "src": "https://example.com/image.jpg",
      "alt": "image description",
      "width": 400,
      "height": 300
    }
  ],
  "videos": [
    {
      "src": "https://example.com/video.mp4", 
      "type": "video",
      "duration": 120
    }
  ],
  "links": [
    {
      "href": "https://example.com/link",
      "text": "link text"
    }
  ],
  "boundingBox": {
    "x": 100,
    "y": 150,
    "width": 300,
    "height": 200
  }
}
```

## Files Structure

- `manifest.json` - Extension manifest (Chrome v3)
- `content.js` - Content script for bounding box functionality
- `content.css` - Styles for the selection overlay
- `popup.html` - Extension popup interface
- `popup.js` - Popup functionality
- `background.js` - Service worker for message handling

## Notes

- This is a barebones implementation focused on core functionality
- The API endpoint is currently a dummy URL that will fail (as intended)
- Console logs show the extracted data for debugging
- No data persistence or complex error handling implemented
- Designed for Chrome/Chromium browsers using Manifest V3
