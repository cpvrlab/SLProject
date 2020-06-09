#pragma once

#include <string>
#include <vector>

#include <Object.h>
#include <GPUShader.h>

using namespace std;

//-----------------------------------------------------------------------------
class GPUProgram : public Object
{

public:
    GPUProgram(string name) : Object(name) { ; }

protected:
    VGPUShader _shaders;    //!< vector of gpu shaders
};
//-----------------------------------------------------------------------------
typedef vector<GPUProgram> VGPUProgram;
//-----------------------------------------------------------------------------
