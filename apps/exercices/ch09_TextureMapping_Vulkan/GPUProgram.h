#ifndef GPUPROGRAM_H
#define GPUPROGRAM_H

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
    ~GPUProgram();

    // Getter
    VGPUShader& shaders() { return _shaders; }

    // Setter
    void addShader(GPUShader* shader);

protected:
    VGPUShader _shaders; //!< vector of gpu shaders
};
//-----------------------------------------------------------------------------
typedef vector<GPUProgram> VGPUProgram;
//-----------------------------------------------------------------------------
#endif
