#include <SLIOBrowserOpen.h>

#ifdef SL_STORAGE_WEB
//-----------------------------------------------------------------------------
SLIOWriterBrowserOpen::SLIOWriterBrowserOpen(SLstring path)
  : SLIOWriterMemory(path)
{
}
//-----------------------------------------------------------------------------
void SLIOWriterBrowserOpen::flush()
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
        let url = URL.createObjectURL(blob);

        let link = document.querySelector("#snapshot-download");
        link.href = url;
        link.download = path;

        document.querySelector("#snapshot-image").src = url;
        document.querySelector("#snapshot-overlay").classList.add("visible");
    }, filename.c_str(), data, length);
    // clang-format on
}
//-----------------------------------------------------------------------------
#endif