//
// Created by nic on 24.10.19.
//

#ifndef SLPROJECT_SLOPTIXHELPER_H
#define SLPROJECT_SLOPTIXHELPER_H

#include <stdafx.h> // Must be the 1st include followed by  an empty line


using namespace std::placeholders;
using namespace std::chrono;


// Optix error-checking and CUDA error-checking are copied from nvidia optix sutil
//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------

#include <optix_stubs.h>

#define OPTIX_CHECK( call )                                                    \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
            throw SLOptixException( res, ss.str().c_str() );                   \
        }                                                                      \
    }


#define OPTIX_CHECK_LOG( call )                                                \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << log                               \
               << ( sizeof_log > sizeof( log ) ? "<TRUNCATED>" : "" )          \
               << "\n";                                                        \
            throw SLOptixException( res, ss.str().c_str() );                   \
        }                                                                      \
    }


//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK( call )                                                     \
    {                                                                          \
        CUresult result = call;                                                \
        if( result != CUDA_SUCCESS )                                           \
        {                                                                      \
            const char *errorstr;                                              \
            cuGetErrorString(result, &errorstr);                               \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << errorstr                                                     \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"                   \
               << result << "\n";                                              \
            throw SLOptixException( ss.str().c_str() );                        \
        }                                                                      \
    }


#define CUDA_SYNC_CHECK( call )                                                \
    {                                                                          \
        CUstream stream = call;                                                \
        CUresult result = cuStreamSynchronize(stream);                         \
        if( result != CUDA_SUCCESS )                                           \
        {                                                                      \
            const char *errorstr;                                              \
            cuGetErrorString(result, &errorstr);                               \
            std::stringstream ss;                                              \
            ss << "CUDA error on synchronize with error '"                     \
               << errorstr                                                     \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw SLOptixException( ss.str().c_str() );                        \
        }                                                                      \
    }

class SLOptixException : public std::runtime_error
{
public:
    SLOptixException( const char* msg )
            : std::runtime_error( msg )
    { }

    SLOptixException( OptixResult res, const char* msg )
            : std::runtime_error( createMessage( res, msg ).c_str() )
    { }

private:
    std::string createMessage( OptixResult res, const char* msg )
    {
        std::ostringstream out;
        out << optixGetErrorName( res ) << ": " << msg;
        return out.str();
    }
};

// Get PTX string from File
std::string getPtxStringFromFile(
        std::string filename,               // Cuda C input file name
        const char** log = NULL );          // (Optional) pointer to compiler log string. If *log == NULL there is no output.

float4 make_float4(const SLVec4f& f);

float3 make_float3(const SLVec3f& f);

#endif //SLPROJECT_SLOPTIXHELPER_H
