//#############################################################################
//  File:      SLOptixCudaBuffer.h
//  Authors:   Nic Dorner
//  Date:      October 2019
//  Authors:   Nic Dorner
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    ifndef SLCUDABUFFER_H
#        define SLCUDABUFFER_H
#        include <cstdio>
#        include <cassert>
#        include <vector>
#        include <cuda.h>
#        include <driver_types.h>
#        include <cuda_runtime_api.h>
#        include <SLOptixHelper.h>

//-----------------------------------------------------------------------------
template<typename T>
class SLOptixCudaBuffer
{
public:
    SLOptixCudaBuffer();
    ~SLOptixCudaBuffer();

    bool isAllocated();

    void resize(size_t);

    //! allocate to given number of bytes
    void alloc(size_t);

    //! free allocated memory
    void free();

    void alloc_and_upload(vector<T>&);
    void alloc_and_upload(T*, unsigned int);

    void upload(vector<T>&);
    void upload(T*);

    void download(T*);

    CUdeviceptr  devicePointer() { return _devicePointer; }
    CUdeviceptr* devicePointerPointer() { return &_devicePointer; }
    size_t       size() { return _size; }

private:
    CUdeviceptr _devicePointer;
    size_t      _size;
};
//-----------------------------------------------------------------------------
template<typename T>
SLOptixCudaBuffer<T>::SLOptixCudaBuffer()
{
    _devicePointer = 0;
    _size          = 0;
}
//-----------------------------------------------------------------------------
template<typename T>
SLOptixCudaBuffer<T>::~SLOptixCudaBuffer()
{
    free();
}
//-----------------------------------------------------------------------------
template<typename T>
bool SLOptixCudaBuffer<T>::isAllocated()
{
    return _devicePointer != 0;
}
//-----------------------------------------------------------------------------
template<typename T>
void SLOptixCudaBuffer<T>::resize(size_t size)
{
    free();
    alloc(size);
}
//-----------------------------------------------------------------------------
template<typename T>
void SLOptixCudaBuffer<T>::alloc(size_t size)
{
    _size = size;
    CUDA_CHECK(cuMemAlloc(&_devicePointer, _size));
}
//-----------------------------------------------------------------------------
template<typename T>
void SLOptixCudaBuffer<T>::free()
{
    if (_devicePointer)
    {
        CUDA_CHECK(cuMemFree(_devicePointer));
        _size          = 0;
        _devicePointer = 0;
    }
}
//-----------------------------------------------------------------------------
template<typename T>
void SLOptixCudaBuffer<T>::alloc_and_upload(vector<T>& vt)
{
    alloc(sizeof(T) * vt.size());
    upload(vt);
}
//-----------------------------------------------------------------------------
template<typename T>
void SLOptixCudaBuffer<T>::alloc_and_upload(T* t, unsigned int count)
{
    alloc(sizeof(T) * count);
    upload(t);
}
//-----------------------------------------------------------------------------
template<typename T>
void SLOptixCudaBuffer<T>::upload(vector<T>& vt)
{
    assert(_size == sizeof(T) * vt.size());
    CUDA_CHECK(cuMemcpyHtoD(_devicePointer,
                            (void*)vt.data(),
                            _size));
}
//-----------------------------------------------------------------------------
template<typename T>
void SLOptixCudaBuffer<T>::upload(T* t)
{
    CUDA_CHECK(cuMemcpyHtoD(_devicePointer,
                            (void*)t,
                            _size));
}
//-----------------------------------------------------------------------------
template<typename T>
void SLOptixCudaBuffer<T>::download(T* t)
{
    assert(_devicePointer != 0);
    CUDA_CHECK(cuMemcpyDtoH((void*)t,
                            _devicePointer,
                            _size));
}
//-----------------------------------------------------------------------------
#    endif // SLCUDABUFFER_H
#endif     // SL_HAS_OPTIX
