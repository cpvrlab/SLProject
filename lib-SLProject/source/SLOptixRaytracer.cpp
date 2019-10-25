//#############################################################################
//  File:      SLOptixRaytracer.cpp
//  Author:    Nic Dorner
//  Date:      October 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################
#include <stdafx.h> // Must be the 1st include followed by  an empty line


using namespace std::placeholders;
using namespace std::chrono;

#include <SLApplication.h>
#include <SLLightRect.h>
#include <SLSceneView.h>
#include <SLOptixRaytracer.h>
#include <SLOptixDefinitions.h>

#include <optix.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <SLOptixHelper.h>

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<CameraData>   RayGenSbtRecord;

//-----------------------------------------------------------------------------
SLOptixRaytracer::SLOptixRaytracer()
{
    name("myCoolRaytracer");

    _state         = rtReady;
    _maxDepth      = 5;

    // set texture properties
    _min_filter   = GL_NEAREST;
    _mag_filter   = GL_NEAREST;
    _wrap_s       = GL_CLAMP_TO_EDGE;
    _wrap_t       = GL_CLAMP_TO_EDGE;
    _resizeToPow2 = false;
}
//-----------------------------------------------------------------------------
SLOptixRaytracer::~SLOptixRaytracer()
{
    SL_LOG("Destructor      : ~SLOptixRaytracer\n");
}

void SLOptixRaytracer::setupOptix() {

    // Set compile options for modules and pipelines
    _module_compile_options = {};
    _module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
    _module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    _module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else
    _module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    _module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    _pipeline_compile_options.usesMotionBlur        = false;
    _pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    _pipeline_compile_options.numPayloadValues      = 2;
    _pipeline_compile_options.numAttributeValues    = 2;
    _pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    _pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

void SLOptixRaytracer::_createContext() {
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &_context ) );
}

void SLOptixRaytracer::_createModule(std::string filename) {
    OptixModule module = nullptr;
    {
        const std::string ptx = getPtxStringFromFile(filename);
        char log[2048];
        size_t sizeof_log = sizeof( log );

        OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                _context,
                &_module_compile_options,
                &_pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module
        ) );
    }
}

void SLOptixRaytracer::_createProgrames() {

}

void SLOptixRaytracer::_createPipeline() {

}
