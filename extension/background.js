chrome.runtime.onInstalled.addListener(() => {
    console.log('FactCheck Selector extension installed');
});
let lastExtraction = null;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "selectionConfirmed") {
    console.log("Selection confirmed", request.data);
    lastExtraction = request.data;

    // Open results window
    chrome.windows.create(
      {
        url: chrome.runtime.getURL("result.html"),
        type: "popup",
        width: 500,
        height: 600,
      },
      (newWindow) => {
        // Send data once window is ready
        setTimeout(() => {
          chrome.runtime.sendMessage({
            action: "displayExtraction",
            data: request.data,
          });
        }, 500);
      }
    );
  } else if (request.action === "getLastExtraction") {
    sendResponse(lastExtraction);
  }
});
