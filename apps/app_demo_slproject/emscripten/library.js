mergeInto(LibraryManager.library, {
    showLoadingOverlay: function (resource) {
        resource = UTF8ToString(resource);
        document.querySelector("#download-text").innerHTML = resource;

        if (globalThis.hideTimer === null) {
            document.querySelector("#overlay").classList.add("visible");
        } else {
            clearTimeout(globalThis.hideTimer);
        }
    },
    hideLoadingOverlay: function () {
        globalThis.hideTimer = setTimeout(function () {
            globalThis.hideTimer = null;
            document.querySelector("#overlay").classList.remove("visible");
        }, 500);
    }
});

