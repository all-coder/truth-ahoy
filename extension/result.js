document.addEventListener("DOMContentLoaded", () => {
  chrome.runtime.onMessage.addListener((request) => {
    if (request.action === "displayExtraction") {
      const data = request.data;
      document.getElementById("stats").textContent =
        `${data.textParagraphs?.length || 0} paragraphs, ` +
        `${data.images?.length || 0} images, ` +
        `${data.videos?.length || 0} videos, ` +
        `${data.links?.length || 0} links`;
      document.getElementById("json").textContent = JSON.stringify(
        data,
        null,
        2
      );
    }
  });

  // Ask background for last extraction if available
  chrome.runtime.sendMessage({ action: "getLastExtraction" }, (data) => {
    if (data) {
      chrome.runtime.sendMessage({ action: "displayExtraction", data });
    }
  });
});
