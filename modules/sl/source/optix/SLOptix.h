//#############################################################################
//  File:      SLOptix.h
//  Authors:   Nic Dorner
//  Date:      October 2019
//  Authors:   Nic Dorner
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    ifndef SLOPTIX_H
#        define SLOPTIX_H
#        include <optix_types.h>
#        include <cuda.h>
#        include <string>
#        include <SLOptixDefinitions.h>

using std::string;

//-----------------------------------------------------------------------------
//! SLOptix base instance for static Optix initialization
class SLOptix
{
public:
    // Public global static Optix objects
    static void               createStreamAndContext();
    static OptixDeviceContext context;
    static CUstream           stream;
    static string             exePath;
};
//-----------------------------------------------------------------------------
#    endif // SLOPTIX_H
#endif     // SL_HAS_OPTIX
