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

#include <utility>
#include <SLCudaBuffer.h>

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
typedef SbtRecord<int>   MissSbtRecord;
typedef SbtRecord<int>   HitSbtRecord;

//-----------------------------------------------------------------------------
SLOptixRaytracer::SLOptixRaytracer()
: SLRaytracer()
{
    name("myCoolRaytracer");

    _params = {};
}
//-----------------------------------------------------------------------------
SLOptixRaytracer::~SLOptixRaytracer()
{
    SL_LOG("Destructor      : ~SLOptixRaytracer\n");

    CUDA_CHECK( cuMemFree( _sbt.raygenRecord       ) );
    CUDA_CHECK( cuMemFree( _sbt.missRecordBase     ) );
    CUDA_CHECK( cuMemFree( _sbt.hitgroupRecordBase ) );

    OPTIX_CHECK( optixPipelineDestroy( _pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( _radiance_hit_group ) );
//    OPTIX_CHECK( optixProgramGroupDestroy( _occlusion_hit_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( _radiance_miss_group ) );
//    OPTIX_CHECK( optixProgramGroupDestroy( _occlusion_miss_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( _raygen_prog_group ) );
    OPTIX_CHECK( optixModuleDestroy( _cameraModule ) );
//    OPTIX_CHECK( optixModuleDestroy( _shadingModule ) );

    OPTIX_CHECK( optixDeviceContextDestroy( _context ) );
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
    _pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_USER;
    _pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    _createContext();

    _cameraModule = _createModule("SLOptixRaytracerCamera.cu");
//    _shadingModule = _createModule("SLOptixRaytracerShading.cu");

    OptixProgramGroupDesc raygen_prog_group_desc  = {}; //
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = _cameraModule;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__draw_solid_color";
    _raygen_prog_group = _createProgram(raygen_prog_group_desc);

    OptixProgramGroupDesc radiance_miss_prog_group_desc = {};
    radiance_miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    _radiance_miss_group = _createProgram(radiance_miss_prog_group_desc);

    OptixProgramGroupDesc radiance_hitgroup_prog_group_desc = {};
    radiance_hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    _radiance_hit_group = _createProgram(radiance_hitgroup_prog_group_desc);

    OptixProgramGroup program_groups[] = { _raygen_prog_group };
    _pipeline = _createPipeline(program_groups, 1);

    _sbt = _createShaderBindingTable();

    _paramsBuffer.alloc(sizeof(Params));
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

void SLOptixRaytracer::_createContext() {
    // Initialize CUDA
    CUDA_CHECK( cuInit( 0 ) );
    CUDA_CHECK( cuMemFree( 0 ) );

    CUcontext          cu_ctx = 0;  // zero means take the current context
    CUcontext           s_ctx = 0;
    CUDA_CHECK( cuCtxCreate( &cu_ctx, 0, 0) );
    CUDA_CHECK( cuStreamCreate( &_stream, CU_STREAM_DEFAULT ) );
    CUDA_CHECK( cuStreamGetCtx(_stream, &s_ctx));
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &_context ) );
}

OptixModule SLOptixRaytracer::_createModule(std::string filename) {
    OptixModule module = nullptr;
    {
        const std::string ptx = getPtxStringFromFile(std::move(filename));
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
    return module;
}

OptixProgramGroup SLOptixRaytracer::_createProgram(OptixProgramGroupDesc prog_group_desc) {
    OptixProgramGroup program_group = {};
    OptixProgramGroupOptions  program_group_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof( log );

    {
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                _context,
                &prog_group_desc,
                1,  // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &program_group
        ) );
    }

    return program_group;
}

OptixPipeline SLOptixRaytracer::_createPipeline(OptixProgramGroup * program_groups, unsigned int numProgramGroups) {

    OptixPipeline pipeline;

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = _maxDepth;
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipeline_link_options.overrideUsesMotionBlur   = false;

    char   log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
            _context,
            &_pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            numProgramGroups,
            log,
            &sizeof_log,
            &pipeline
    ) );

    return pipeline;
}

OptixShaderBindingTable SLOptixRaytracer::_createShaderBindingTable() {

    OptixShaderBindingTable sbt = {};
    {
        std::vector<RayGenSbtRecord> rayGenRecords;
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( _raygen_prog_group, &rg_sbt ) );
        rg_sbt.data = {1.0f, 0.f, 0.f};
        SLCudaBuffer<RayGenSbtRecord> rayGenBuffer = SLCudaBuffer<RayGenSbtRecord>();
        rayGenRecords.push_back(rg_sbt);
        rayGenBuffer.alloc_and_upload(rayGenRecords);

        std::vector<MissSbtRecord> missRecords;
        MissSbtRecord ms_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( _radiance_miss_group , &ms_sbt ) );
        SLCudaBuffer<MissSbtRecord> missBuffer = SLCudaBuffer<MissSbtRecord>();
        missRecords.push_back(ms_sbt);
        missBuffer.alloc_and_upload(missRecords);

        std::vector<HitSbtRecord> hitRecords;
        HitSbtRecord hg_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( _radiance_hit_group, &hg_sbt ) );
        SLCudaBuffer<HitSbtRecord> hitBuffer = SLCudaBuffer<HitSbtRecord>();
        hitRecords.push_back(hg_sbt);
        hitBuffer.alloc_and_upload(hitRecords);

        sbt.raygenRecord                = rayGenBuffer.devicePointer();
        sbt.missRecordBase              = missBuffer.devicePointer();
        sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
        sbt.missRecordCount             = 1;
        sbt.hitgroupRecordBase          = hitBuffer.devicePointer();
        sbt.hitgroupRecordStrideInBytes = sizeof( HitSbtRecord );
        sbt.hitgroupRecordCount         = 1;
    }

    return sbt;
}

void SLOptixRaytracer::setupScene(SLSceneView *sv) {
    _sv = sv;

    _imageBuffer.resize(_sv->scrW() * _sv->scrH() * sizeof(uchar3));

    _params.image = reinterpret_cast<uchar3 *>(_imageBuffer.devicePointer());
    _params.image_width = _sv->scrW();

    _paramsBuffer.upload(&_params);
}

SLbool SLOptixRaytracer::renderClassic() {
    CUcontext ctx;
    CUDA_CHECK( cuCtxGetCurrent(&ctx));

    OPTIX_CHECK( optixLaunch(
            _pipeline,
            _stream,
            _paramsBuffer.devicePointer(),
            _paramsBuffer.size(),
            &_sbt,
            _sv->scrW(),
            _sv->scrH(),
            /*depth=*/1 ) );
    CUDA_SYNC_CHECK(_stream);

    prepareImage();       // Setup image & precalculations

    uchar3 image[_sv->scrW() * _sv->scrH()];
    _imageBuffer.download(image);;

    _images[0]->load(
            _sv->scrW(),
            _sv->scrH(),
            PF_rgb,
            PF_rgb,
            reinterpret_cast<uchar *>(image),
            true,
            true);
}
