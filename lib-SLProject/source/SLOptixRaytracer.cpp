//#############################################################################
//  File:      SLOptixRaytracer.cpp
//  Author:    Nic Dorner
//  Date:      October 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################
#include <SLApplication.h>
#include <SLLightRect.h>
#include <SLSceneView.h>
#include <SLOptixRaytracer.h>
#include <SLOptixDefinitions.h>

#include <optix.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <utility>
#include <optix_stack_size.h>

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

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

    OPTIX_CHECK( optixPipelineDestroy( _pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( _radiance_hit_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( _occlusion_hit_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( _radiance_miss_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( _occlusion_miss_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( _raygen_prog_group ) );
    OPTIX_CHECK( optixModuleDestroy( _cameraModule ) );
    OPTIX_CHECK( optixModuleDestroy( _shadingModule ) );

    OPTIX_CHECK( optixDeviceContextDestroy( _context ) );
}

void SLOptixRaytracer::setupOptix() {

    // Set compile options for modules and pipelines
    _module_compile_options = {};
    _module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
    _module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    _module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    _pipeline_compile_options.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONEE;
#else
    _module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    _module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    _pipeline_compile_options.exceptionFlags     = OPTIX_EXCEPTION_FLAG_USER;
#endif

    _pipeline_compile_options.usesMotionBlur        = false;
    _pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    _pipeline_compile_options.numPayloadValues      = 7;
    _pipeline_compile_options.numAttributeValues    = 2;
    _pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    _createContext();

    _cameraModule = _createModule("SLOptixRaytracerCamera.cu");
    _shadingModule = _createModule("SLOptixRaytracerShading.cu");

    OptixProgramGroupDesc raygen_prog_group_desc  = {}; //
    raygen_prog_group_desc.kind                                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module                            = _cameraModule;
    raygen_prog_group_desc.raygen.entryFunctionName                 = "__raygen__pinhole_camera";
    _raygen_prog_group = _createProgram(raygen_prog_group_desc);

    OptixProgramGroupDesc radiance_miss_prog_group_desc = {};
    radiance_miss_prog_group_desc.kind                              = OPTIX_PROGRAM_GROUP_KIND_MISS;
    radiance_miss_prog_group_desc.miss.module                       = _shadingModule;
    radiance_miss_prog_group_desc.miss.entryFunctionName            = "__miss__radiance";
    _radiance_miss_group = _createProgram(radiance_miss_prog_group_desc);

    OptixProgramGroupDesc occlusion_miss_prog_group_desc = {};
    occlusion_miss_prog_group_desc.kind                             = OPTIX_PROGRAM_GROUP_KIND_MISS;
    occlusion_miss_prog_group_desc.miss.module                      = _shadingModule;
    occlusion_miss_prog_group_desc.miss.entryFunctionName           = "__miss__occlusion";
    _occlusion_miss_group = _createProgram(occlusion_miss_prog_group_desc);

    OptixProgramGroupDesc radiance_hitgroup_prog_group_desc = {};
    radiance_hitgroup_prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_hitgroup_prog_group_desc.hitgroup.moduleAH             = _shadingModule;
    radiance_hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH  = "__anyhit__radiance";
    radiance_hitgroup_prog_group_desc.hitgroup.moduleCH             = _shadingModule;
    radiance_hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH  = "__closesthit__radiance";
    _radiance_hit_group = _createProgram(radiance_hitgroup_prog_group_desc);

    OptixProgramGroupDesc occlusion_hitgroup_prog_group_desc = {};
    occlusion_hitgroup_prog_group_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_hitgroup_prog_group_desc.hitgroup.moduleAH             = _shadingModule;
    occlusion_hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH  = "__anyhit__occlusion";
    _occlusion_hit_group = _createProgram(occlusion_hitgroup_prog_group_desc);

    OptixProgramGroup program_groups[] = {
            _raygen_prog_group,
            _radiance_miss_group,
            _occlusion_miss_group,
            _radiance_hit_group,
            _occlusion_hit_group,
    };
    _pipeline = _createPipeline(program_groups, 5);

    _paramsBuffer.alloc(sizeof(Params));
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

void SLOptixRaytracer::_createContext() {
    // Initialize CUDA
    CUcontext        cu_ctx = 0;  // zero means take the current context
    CUDA_CHECK( cuInit( 0 ) );
    CUDA_CHECK( cuMemFree( 0 ) );
    CUDA_CHECK( cuCtxCreate( &cu_ctx, 0, 0) );
    CUDA_CHECK( cuStreamCreate( &_stream, CU_STREAM_DEFAULT ) );

    // Initialize OptiX
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

OptixShaderBindingTable SLOptixRaytracer::_createShaderBindingTable(SLVMesh meshes) {

    OptixShaderBindingTable sbt = {};
    {
        // Setup ray generation records
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( _raygen_prog_group, &rg_sbt ) );
        _rayGenBuffer.alloc_and_upload(&rg_sbt, 1);

        // Setup miss records
        std::vector<MissSbtRecord> missRecords;

        MissSbtRecord radiance_ms_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( _radiance_miss_group , &radiance_ms_sbt ) );
        radiance_ms_sbt.data.bg_color = {1.0f, 1.0f, 1.0f, 1.0f};
        missRecords.push_back(radiance_ms_sbt);

        MissSbtRecord occlusion_ms_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( _occlusion_miss_group , &occlusion_ms_sbt ) );
        missRecords.push_back(occlusion_ms_sbt);

        _missBuffer.alloc_and_upload(missRecords);

        // Setup hit records
        std::vector<HitSbtRecord> hitRecords;

        for(auto mesh : meshes) {
            HitSbtRecord radiance_hg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( _radiance_hit_group, &radiance_hg_sbt ) );
            radiance_hg_sbt.data.sbtIndex = 0;
//            radiance_hg_sbt.data.normals = mesh->N.data();
//            radiance_hg_sbt.data.indices = mesh->N.data();
            radiance_hg_sbt.data.material.kn = mesh->mat()->kn();
            radiance_hg_sbt.data.material.kt = mesh->mat()->kt();
            radiance_hg_sbt.data.material.kr = mesh->mat()->kr();
            radiance_hg_sbt.data.material.shininess = mesh->mat()->shininess();
            radiance_hg_sbt.data.material.ambient_color = make_float4(mesh->mat()->ambient());
            radiance_hg_sbt.data.material.specular_color = make_float4(mesh->mat()->specular());
            radiance_hg_sbt.data.material.diffuse_color = make_float4(mesh->mat()->diffuse());
            hitRecords.push_back(radiance_hg_sbt);

            HitSbtRecord occlusion_hg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( _occlusion_hit_group, &occlusion_hg_sbt ) );
            occlusion_hg_sbt.data.material.kt = mesh->mat()->kt();
            hitRecords.push_back(occlusion_hg_sbt);

        }
        _hitBuffer.alloc_and_upload(hitRecords);

        sbt.raygenRecord                = _rayGenBuffer.devicePointer();
        sbt.missRecordBase              = _missBuffer.devicePointer();
        sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
        sbt.missRecordCount             = RAY_TYPE_COUNT;
        sbt.hitgroupRecordBase          = _hitBuffer.devicePointer();
        sbt.hitgroupRecordStrideInBytes = sizeof( HitSbtRecord );
        sbt.hitgroupRecordCount         = RAY_TYPE_COUNT * meshes.size();
    }

    return sbt;
}

void SLOptixRaytracer::_buildAccelerationStructure(OptixBuildInput buildInput, SLOptixAccelerationStructure* accelerationStructure)
{
    OptixTraversableHandle handle;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes accel_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
            _context,
            &accel_options,
            &buildInput,
            1,  // num_build_inputs
            &accel_buffer_sizes
    ) );

    SLCudaBuffer<void> temp_buffer = SLCudaBuffer<void>();
    temp_buffer.alloc(accel_buffer_sizes.tempSizeInBytes);

    // non-compacted output
    auto* non_compacted_output_buffer = new SLCudaBuffer<void>();
    non_compacted_output_buffer->alloc(accel_buffer_sizes.outputSizeInBytes);

    SLCudaBuffer<OptixAabb> aabbBuffer = SLCudaBuffer<OptixAabb>();
    aabbBuffer.alloc(sizeof(OptixAabb));
    SLCudaBuffer<size_t> compactedSize = SLCudaBuffer<size_t>();
    compactedSize.alloc(sizeof(size_t));

    OptixAccelEmitDesc emitProperty[2];
    emitProperty[0].type               = OPTIX_PROPERTY_TYPE_AABBS;
    emitProperty[0].result             = aabbBuffer.devicePointer();
    emitProperty[1].type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty[1].result             = compactedSize.devicePointer();

    OPTIX_CHECK( optixAccelBuild(
            _context,
            _stream,                            // CUDA stream
            &accel_options,
            &buildInput,
            1,                                  // num build inputs
            temp_buffer.devicePointer(),
            accel_buffer_sizes.tempSizeInBytes,
            non_compacted_output_buffer->devicePointer(),
            accel_buffer_sizes.outputSizeInBytes,
            &handle,
            emitProperty,                      // emitted property list
            2                                   // num emitted properties
    ) );
    CUDA_SYNC_CHECK( _stream );

    OptixAabb aabb;
    aabbBuffer.download(&aabb);

    size_t compacted_accel_size;
    compactedSize.download(&compacted_accel_size);

    if(compacted_accel_size < accel_buffer_sizes.outputSizeInBytes )
    {
        auto* outputBuffer = new SLCudaBuffer<void>();
        outputBuffer->alloc(compacted_accel_size );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact(_context,
                _stream,
                handle,
                outputBuffer->devicePointer(),
                compacted_accel_size,
                &handle ) );
        CUDA_SYNC_CHECK( _stream );

        delete non_compacted_output_buffer;
        accelerationStructure->buffer(outputBuffer);
    } else {
        accelerationStructure->buffer(non_compacted_output_buffer);
    }

    accelerationStructure->optixTraversableHandle(handle);
}

void SLOptixRaytracer::_createMeshAccelerationStructure(SLMesh* mesh, unsigned int index)
{
    //
    // copy mesh data to device
    //
    SLCudaBuffer<SLVec3f> vertexBuffer = SLCudaBuffer<SLVec3f>();
    vertexBuffer.alloc_and_upload(mesh->P);

    //
    // Build triangle GAS
    //
    uint32_t triangle_input_flags[1] =  // One per SBT record for this build input
    {
        OPTIX_GEOMETRY_FLAG_NONE
    };

    OptixBuildInput triangle_input                           = {};
    triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    SLCudaBuffer<SLuint> indexBufferInt = SLCudaBuffer<SLuint>();
    SLCudaBuffer<SLushort> indexBufferShort = SLCudaBuffer<SLushort>();
    if (mesh->I16.size() < USHRT_MAX) {
        indexBufferShort.alloc_and_upload(mesh->I16);

        triangle_input.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
        triangle_input.triangleArray.numIndexTriplets            = mesh->I16.size() / 3;
        triangle_input.triangleArray.indexBuffer                 = indexBufferShort.devicePointer();
    } else {
        indexBufferInt.alloc_and_upload(mesh->I32);

        triangle_input.triangleArray.indexFormat                 = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.numIndexTriplets            = mesh->I32.size() / 3;
        triangle_input.triangleArray.indexBuffer                 = indexBufferInt.devicePointer();
    }

    triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexBuffers               = vertexBuffer.devicePointerPointer();
    triangle_input.triangleArray.numVertices                 = mesh->P.size();
    triangle_input.triangleArray.flags                       = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords               = 1;
//    triangle_input.triangleArray.sbtIndexOffsetBuffer        = 0;
//    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
//    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

    mesh->sbtIndex(RAY_TYPE_COUNT * index);

     _buildAccelerationStructure(triangle_input, mesh);
}

void SLOptixRaytracer::_createInstanceAccelerationStructure(SLNode* node) {
    std::vector<OptixInstance> instances;

    SLVNode children = node->children();
    unsigned int i;
    for(auto child : children) {
        _createInstanceAccelerationStructure(child);

        if (child->optixTraversableHandle()) {
            OptixInstance instance;

            const SLMat4f mat4x4 = child->updateAndGetWM();
            float transform[12] = {mat4x4.m(0), mat4x4.m(4), mat4x4.m(8), mat4x4.m(12),
                                   mat4x4.m(1), mat4x4.m(5), mat4x4.m(9), mat4x4.m(13),
                                   mat4x4.m(2), mat4x4.m(6), mat4x4.m(10), mat4x4.m(14)};
            memcpy(instance.transform, transform, sizeof(float)*12);

            instance.instanceId = i++;
            instance.visibilityMask = 255;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = child->optixTraversableHandle();
            instance.sbtOffset = 0;

            instances.push_back(instance);
        }
    }

    SLVMesh meshes = node->meshes();
    for(auto mesh : meshes) {
        OptixInstance instance;

        const SLMat4f& mat4x4 = node->updateAndGetWM();
        float transform[12] = {mat4x4.m(0), mat4x4.m(4), mat4x4.m(8), mat4x4.m(12),
                               mat4x4.m(1), mat4x4.m(5), mat4x4.m(9), mat4x4.m(13),
                               mat4x4.m(2), mat4x4.m(6), mat4x4.m(10), mat4x4.m(14)};
        memcpy(instance.transform, transform, sizeof(float)*12);

        instance.instanceId = 0;
        instance.visibilityMask = 255;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = mesh->optixTraversableHandle();
        instance.sbtOffset = mesh->sbtIndex();

        instances.push_back(instance);
    }

    if (instances.empty()) {
        node->optixTraversableHandle(0);
        return;
    }

    SLCudaBuffer<OptixInstance> instanceBuffer = SLCudaBuffer<OptixInstance>();
    instanceBuffer.alloc_and_upload(instances);

    OptixBuildInput instance_input                           = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = instanceBuffer.devicePointer();
    instance_input.instanceArray.numInstances = instances.size();

    _buildAccelerationStructure(instance_input, node);
}

void SLOptixRaytracer::setupScene() {
    SLScene* scene = SLApplication::scene;
    SLVMesh meshes = scene->meshes();

    // Iterate over all meshes
    unsigned int idx = 0;
    for(auto mesh : meshes) {
        _createMeshAccelerationStructure(mesh, idx++);
    }

    _createInstanceAccelerationStructure(scene->root3D());

    _sbt = _createShaderBindingTable(meshes);

    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes( _raygen_prog_group,    &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( _radiance_miss_group,  &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( _occlusion_miss_group, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( _radiance_hit_group,   &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( _occlusion_hit_group,  &stack_sizes ) );
    unsigned int directCallableStackSizeFromTraversal;
    unsigned int directCallableStackSizeFromState;
    unsigned int continuationStackSize;
    OPTIX_CHECK( optixUtilComputeStackSizes(&stack_sizes,
            1,
            0,
            0,
            &directCallableStackSizeFromTraversal,
            &directCallableStackSizeFromState,
            &continuationStackSize) );
    OPTIX_CHECK( optixPipelineSetStackSize(_pipeline,
                              directCallableStackSizeFromTraversal,
                              directCallableStackSizeFromState,
                              continuationStackSize,
                              scene->maxTreeDepth()) );
}

void SLOptixRaytracer::updateScene(SLSceneView *sv) {
    SLScene* scene = SLApplication::scene;
    SLCamera* camera = sv->camera();
    _sv = sv;

    RayGenSbtRecord rayGenSbtRecord;
    _rayGenBuffer.download(&rayGenSbtRecord);
    SLVec3f eye, u, v, w;
    camera->UVWFrame(eye, u, v, w);
    rayGenSbtRecord.data.eye = make_float3(eye);
    rayGenSbtRecord.data.U = make_float3(u);
    rayGenSbtRecord.data.V = make_float3(v);
    rayGenSbtRecord.data.W = make_float3(w);
    _rayGenBuffer.upload(&rayGenSbtRecord);

    _imageBuffer.resize(_sv->scrW() * _sv->scrH() * sizeof(uchar4));
    _debugBuffer.resize(_sv->scrW() * _sv->scrH() * sizeof(float3));

    _params.image = reinterpret_cast<uchar4 *>(_imageBuffer.devicePointer());
    _params.debug = reinterpret_cast<float3 *>(_debugBuffer.devicePointer());
    _params.width = _sv->scrW();
    _params.height = _sv->scrH();
    _params.handle = scene->root3D()->optixTraversableHandle();
    _params.max_depth = _maxDepth;

    std::vector<Light> lights;
    _lightBuffer.free();
    for(auto light : scene->lights()) {
        SLVec3f position = { light->positionWS().x, light->positionWS().y, light->positionWS().z};
        lights.push_back({
           make_float3(position),
           make_float4(light->diffuse())
        });
    }
    _lightBuffer.alloc_and_upload(lights);
    _params.lights = reinterpret_cast<Light *>(_lightBuffer.devicePointer());
    _params.numLights = lights.size();

    _paramsBuffer.upload(&_params);
}

SLbool SLOptixRaytracer::renderClassic() {
    _state      = rtBusy; // From here we state the RT as busy
    _pcRendered = 0;      // % rendered
    _renderSec  = 0.0f;   // reset time

    initStats(_maxDepth); // init statistics
    prepareImage();       // Setup image & precalculations

    // Measure time
    double t1     = SLApplication::timeS();
    double tStart = t1;

    OPTIX_CHECK(optixLaunch(
            _pipeline,
            _stream,
            _paramsBuffer.devicePointer(),
            _paramsBuffer.size(),
            &_sbt,
            _sv->scrW(),
            _sv->scrH(),
            /*depth=*/1));
    CUDA_SYNC_CHECK(_stream);

    _renderSec  = (SLfloat)(SLApplication::timeS() - tStart);
    _pcRendered = 100;

    _state = rtReady;
    return true;
}

void SLOptixRaytracer::renderImage() {
    SLGLTexture::bindActive(0);

    CUarray texture_ptr;
    CUDA_CHECK( cuGraphicsMapResources(1, &_cudaGraphicsResource, _stream) );
    CUDA_CHECK( cuGraphicsSubResourceGetMappedArray(&texture_ptr, _cudaGraphicsResource, 0, 0) );

    CUDA_ARRAY_DESCRIPTOR des;
    cuArrayGetDescriptor(&des, texture_ptr);
    CUDA_MEMCPY2D memcpy2D;
    memcpy2D.srcDevice = _imageBuffer.devicePointer();
    memcpy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    memcpy2D.srcXInBytes = 0;
    memcpy2D.srcY = 0;
    memcpy2D.srcPitch = 0;
    memcpy2D.dstArray = texture_ptr;
    memcpy2D.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    memcpy2D.dstXInBytes = 0;
    memcpy2D.dstY = 0;
    memcpy2D.dstPitch = 0;
    memcpy2D.WidthInBytes = des.Width * des.NumChannels;
    memcpy2D.Height = des.Height;
    CUDA_CHECK(cuMemcpy2D(&memcpy2D) );

    CUDA_CHECK( cuGraphicsUnmapResources(1, &_cudaGraphicsResource, _stream) );

    SLfloat w = (SLfloat)_sv->scrW();
    SLfloat h = (SLfloat)_sv->scrH();
    if (Utils::abs(_images[0]->width() - w) > 0.0001f) return;
    if (Utils::abs(_images[0]->height() - h) > 0.0001f) return;

    // Set orthographic projection with the size of the window
    SLGLState* stateGL = SLGLState::instance();
    stateGL->projectionMatrix.ortho(0.0f, w, 0.0f, h, -1.0f, 0.0f);
    stateGL->modelViewMatrix.identity();
    stateGL->clearColorBuffer();
    stateGL->depthTest(false);
    stateGL->multiSample(false);
    stateGL->polygonLine(false);

    drawSprite(false);

    stateGL->depthTest(true);
    GET_GL_ERROR;

    float3* debug = reinterpret_cast<float3 *>( malloc(_debugBuffer.size()) );
    _debugBuffer.download(debug);;
    free(debug);
}
