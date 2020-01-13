//#############################################################################
//  File:      SLOptixDefinitions.h
//  Author:    Nic Dorner
//  Date:      October 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    ifndef SLOPTIXACCELERATIONSTRUCTURE_H
#        define SLOPTIXACCELERATIONSTRUCTURE_H
#        include <optix_types.h>
#        include <SLCudaBuffer.h>

//------------------------------------------------------------------------------
class SLOptixAccelerationStructure
{
    public:
    SLOptixAccelerationStructure();
    ~SLOptixAccelerationStructure();

    OptixTraversableHandle optixTraversableHandle() { return _handle; }

    protected:
    void buildAccelerationStructure();
    void updateAccelerationStructure();

    OptixBuildInput        _buildInput        = {};
    OptixAccelBuildOptions _accelBuildOptions = {};
    OptixAccelBufferSizes  _accelBufferSizes  = {};
    OptixTraversableHandle _handle            = 0; //!< Handle for generated geometry acceleration structure
    SLCudaBuffer<void>*    _buffer;
};
//------------------------------------------------------------------------------
#    endif // SLOPTIXACCELERATIONSTRUCTURE_H
#endif     // SL_HAS_OPTIX
