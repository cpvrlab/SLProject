#include <SLAssimpIOSystem.h>

//-----------------------------------------------------------------------------
SLAssimpIOStream::SLAssimpIOStream(SLIOStream* stream)
  : _stream(stream)
{
}
//-----------------------------------------------------------------------------
size_t SLAssimpIOStream::Read(void* pvBuffer, size_t pSize, size_t pCount)
{
    char*  dest      = (char*)pvBuffer;
    size_t totalSize = pCount * pSize;
    size_t sizeRead  = _stream->read(dest, totalSize);
    return sizeRead / pSize;
}
//-----------------------------------------------------------------------------
size_t SLAssimpIOStream::Write(const void* pvBuffer, size_t pSize, size_t pCount)
{
    const char* dest      = (const char*)pvBuffer;
    size_t      totalSize = pCount * pSize;
    return _stream->write(dest, totalSize);
}
//-----------------------------------------------------------------------------
aiReturn SLAssimpIOStream::Seek(size_t pOffset, aiOrigin pOrigin)
{
    SLIOStream::Origin streamOrigin = SLIOStream::IOO_beg;
    if (pOrigin == aiOrigin_SET)
        streamOrigin = SLIOStream::IOO_beg;
    else if (pOrigin == aiOrigin_CUR)
        streamOrigin = SLIOStream::IOO_cur;
    else if (pOrigin == aiOrigin_END)
        streamOrigin = SLIOStream::IOO_end;

    bool successful = _stream->seek(pOffset, streamOrigin);
    return successful ? aiReturn_SUCCESS : aiReturn_FAILURE;
}
//-----------------------------------------------------------------------------
size_t SLAssimpIOStream::Tell() const
{
    return _stream->tell();
}
//-----------------------------------------------------------------------------
size_t SLAssimpIOStream::FileSize() const
{
    return _stream->size();
}
//-----------------------------------------------------------------------------
void SLAssimpIOStream::Flush()
{
    _stream->flush();
}
//-----------------------------------------------------------------------------
bool SLAssimpIOSystem::Exists(const char* pFile) const
{
    return SLFileStorage::exists(pFile, IOK_model);
}
//-----------------------------------------------------------------------------
char SLAssimpIOSystem::getOsSeparator() const
{
    return '/';
}
//-----------------------------------------------------------------------------
Assimp::IOStream* SLAssimpIOSystem::Open(const char* pFile, const char* pMode)
{
    // Assimp requires the modes "wb", "w", "wt", "rb", "r" and "rt".
    // The second character is ignored because SLIOStreams are always binary.
    // Therefor we only need to check the first character to determine
    // whether we are reading or writing.
    SLIOStreamMode streamMode = IOM_read;
    if (pMode[0] == 'r')
        streamMode = IOM_read;
    else if (pMode[0] == 'w')
        streamMode = IOM_write;

    SLIOStream* stream = SLFileStorage::open(pFile, IOK_model, streamMode);
    return new SLAssimpIOStream(stream);
}
//-----------------------------------------------------------------------------
void SLAssimpIOSystem::Close(Assimp::IOStream* pFile)
{
    SLAssimpIOStream* stream = dynamic_cast<SLAssimpIOStream*>(pFile);
    SLFileStorage::close(stream->stream());
    delete stream;
}
//-----------------------------------------------------------------------------