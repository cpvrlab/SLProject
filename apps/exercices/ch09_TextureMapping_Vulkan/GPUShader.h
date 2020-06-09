#pragma once

#include <string>
#include <vector>

#include <Object.h>

using namespace std;

class GPUShader : public Object
{

public:
    GPUShader(string GPUShader) : Object(GPUShader) { ; }

protected:
    string _code;       //!< string of the shader source code
    string _filename;   //!< path and filename of the shader source code
};
//-----------------------------------------------------------------------------
typedef vector<GPUShader> VGPUShader;
//-----------------------------------------------------------------------------
