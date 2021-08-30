//#############################################################################
//  File:      SLOptixAccelStruct.h
//  Authors:   Nic Dorner
//  Date:      October 2019
//  Authors:   Nic Dorner
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    ifndef SLOPTIXACCELSTRUCT_H
#        define SLOPTIXACCELSTRUCT_H
#        include <optix_types.h>
#        include <SLOptixCudaBuffer.h>

//------------------------------------------------------------------------------
class SLOptixAccelStruct
{
public:
    SLOptixAccelStruct();
    ~SLOptixAccelStruct();

    OptixTraversableHandle optixTraversableHandle() { return _handle; }

protected:
    void buildAccelerationStructure();
    void updateAccelerationStructure();

    OptixBuildInput          _buildInput        = {};
    OptixAccelBuildOptions   _accelBuildOptions = {};
    OptixAccelBufferSizes    _accelBufferSizes  = {};
    OptixTraversableHandle   _handle            = 0; //!< Handle for generated geometry acceleration structure
    SLOptixCudaBuffer<void>* _buffer;
};
//------------------------------------------------------------------------------
#    endif // SLOPTIXACCELSTRUCT_H
#endif     // SL_HAS_OPTIX
