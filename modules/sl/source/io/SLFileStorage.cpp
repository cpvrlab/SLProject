#include <SLFileStorage.h>

#if defined(SL_STORAGE_FS)
#    include <SLIONative.h>
#elif defined(SL_STORAGE_WEB)
#    include <SLIOFetch.h>
#    include <SLIOMemory.h>
#    include <SLIONative.h>
#endif

//-----------------------------------------------------------------------------
SLIOStream* SLFileStorage::open(SLstring       path,
                                SLIOStreamKind kind,
                                SLIOStreamMode mode)
{
#if defined(SL_STORAGE_FS)
    if (mode == IOM_read)
        return new SLIOReaderNative(path);
    else if (mode == IOM_write)
        return new SLIOWriterNative(path);
    else
        return nullptr;
#elif defined(SL_STORAGE_WEB)
    SL_LOG("OPENING \"%s\", (%d)", path.c_str(), kind);

    if (mode == IOM_read)
    {
        if (kind == IOK_shader)
        {
            if (SLIOMemory::exists(path))
                return new SLIOReaderMemory(path);
            else
                return new SLIOReaderFetch(path);
        }
        else if (kind == IOK_image || kind == IOK_model)
            return new SLIOReaderFetch(path);
        else
            return new SLIOReaderNative(path);
    }
    else if (mode == IOM_write)
        return new SLIOReaderMemory(path);

    return nullptr;
#endif
}
//-----------------------------------------------------------------------------
void SLFileStorage::close(SLIOStream* stream)
{
    delete stream;
}
//-----------------------------------------------------------------------------
bool SLFileStorage::exists(SLstring path, SLIOStreamKind kind)
{
#if defined(SL_STORAGE_FS)
    return Utils::fileExists(path);
#elif defined(SL_STORAGE_WEB)
    if (path == "")
        return false;

    if (kind == IOK_shader)
        return SLIOMemory::exists(path) || SLIOReaderFetch::exists(path);
    else
        return SLIOReaderFetch::exists(path);
#endif
}
//-----------------------------------------------------------------------------
SLIOBuffer SLFileStorage::readIntoBuffer(SLstring path, SLIOStreamKind kind)
{
    SLIOStream*    stream = open(path, kind, IOM_read);
    size_t         size   = stream->size();
    unsigned char* data   = new unsigned char[size];
    stream->read(data, size);
    close(stream);

    return SLIOBuffer{data, size};
}
//-----------------------------------------------------------------------------
void SLFileStorage::deleteBuffer(SLIOBuffer& buffer)
{
    delete[] buffer.data;
}
//-----------------------------------------------------------------------------