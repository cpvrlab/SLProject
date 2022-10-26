#include <SLIOBrowserDisplay.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
SLIOWriterBrowserDisplay::SLIOWriterBrowserDisplay(SLstring path)
  : SLIOWriterMemory(path)
{
}
//-----------------------------------------------------------------------------
SLIOWriterBrowserDisplay::~SLIOWriterBrowserDisplay()
{
    SLIOMemory::clear(_path);
}
//-----------------------------------------------------------------------------
void SLIOWriterBrowserDisplay::flush()
{
    std::vector<char>& buffer = SLIOMemory::get(_path);
    const char*        data   = buffer.data();
    size_t             length = buffer.size();

    SLstring filename = Utils::getFileName(_path);

    // clang-format off
    MAIN_THREAD_EM_ASM({
        let path = UTF8ToString($0);
        let section = HEAPU8.subarray($1, $1 + $2);
        let array = new Uint8Array(section);
        let blob = new Blob([array], {"type": "image/png"});
        globalThis.snapshotURL = URL.createObjectURL(blob);

        let link = document.querySelector("#snapshot-download");
        link.href = snapshotURL;
        link.download = path;

        document.querySelector("#snapshot-image").src = snapshotURL;
        document.querySelector("#snapshot-overlay").classList.add("visible");
    }, filename.c_str(), data, length);
    // clang-format on
}
//-----------------------------------------------------------------------------
#endif