//#############################################################################
//  File:      SLOptix.h
//  Author:    Nic Dorner
//  Date:      October 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    ifndef SLOPTIX_H
#        define SLOPTIX_H
#        include <optix_types.h>
#        include <cuda.h>
#        include <SLOptixDefinitions.h>

//-----------------------------------------------------------------------------
//! SLOptix base instance for static Optix initialization
class SLOptix
{
public:
    // Public global static Optix objects
    static void               createStreamAndContext();
    static OptixDeviceContext context;
    static CUstream           stream;
};
//-----------------------------------------------------------------------------
// Global static methods
//-----------------------------------------------------------------------------
//! callback function for optix
void context_log_cb(unsigned int level,
                    const char*  tag,
                    const char*  message,
                    void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}
//-----------------------------------------------------------------------------
//! creates the optix and cuda context for the application
void SLOptix::createStreamAndContext()
{
    // Initialize CUDA
    CUcontext cu_ctx;
    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuMemFree(0));
    CUDA_CHECK(cuCtxCreate(&cu_ctx, 0, 0));
    CUDA_CHECK(cuStreamCreate(&SLOptixRaytracer::stream,
                              CU_STREAM_DEFAULT));

    // Initialize OptiX
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx,
                                         &options,
                                         &SLOptixRaytracer::context));
}
//-----------------------------------------------------------------------------
#    endif // SLOPTIX_H
#endif     // SL_HAS_OPTIX
