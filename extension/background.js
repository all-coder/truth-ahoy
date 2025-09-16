chrome.runtime.onInstalled.addListener(() => {
    console.log('FactCheck Selector extension installed');
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'selectionConfirmed') {
        console.log('Selection confirmed in tab:', sender.tab.id);
        console.log('Extracted data:', request.data);

        // Forward message to popup if it's open
        chrome.runtime.sendMessage(request).catch(() => {
            // Popup might not be open
        });
    } else if (request.action === 'selectionCancelled') {
        console.log('Selection cancelled in tab:', sender.tab.id);

        // Forward message to popup if it's open
        chrome.runtime.sendMessage(request).catch(() => {
            // Popup might not be open
        });
    }
});