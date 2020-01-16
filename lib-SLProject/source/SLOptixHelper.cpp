//#############################################################################
//  File:      SLOptixHelper.cpp
//  Author:    Nic Dorner
//  Date:      October 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef SL_HAS_OPTIX
#    include <optix_function_table_definition.h>
#    include <string>
#    include <fstream>
#    include <sstream>
#    include <cuda_runtime.h>
#    include <SLOptixHelper.h>

//-----------------------------------------------------------------------------
static bool readSourceFile(std::string& str, const std::string& filename)
{
    // Try to open file
    std::ifstream file(filename.c_str());
    if (file.good())
    {
        // Found usable source file
        std::stringstream source_buffer;
        source_buffer << file.rdbuf();
        str = source_buffer.str();
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
static std::string getPtxFilename(
  std::string filename)
{
    std::string ptxFilename;
    ptxFilename += '/';
    ptxFilename += "cuda_compile_ptx_1";
    ptxFilename += "_generated_";
    ptxFilename += filename;
    ptxFilename += ".ptx";
    return ptxFilename;
}
//-----------------------------------------------------------------------------
std::string getPtxStringFromFile(
  std::string  filename,
  const char** log)
{
    if (log)
        *log = NULL;

    std::string *ptx, sourceFilePath;
    sourceFilePath = "lib-SLProject/" + getPtxFilename(filename);

    ptx = new std::string();

    // Try to open source PTX file
    if (!readSourceFile(*ptx, sourceFilePath))
    {
        std::string err = "Couldn't open source file " + sourceFilePath;
        throw std::runtime_error(err.c_str());
    }

    return *ptx;
}
//-----------------------------------------------------------------------------
float4 make_float4(const SLVec4f& f)
{
    return {f.x, f.y, f.z, f.w};
};
//-----------------------------------------------------------------------------
float3 make_float3(const SLVec3f& f)
{
    return {f.x, f.y, f.z};
};
    //-----------------------------------------------------------------------------

#endif // SL_HAS_OPTIX
