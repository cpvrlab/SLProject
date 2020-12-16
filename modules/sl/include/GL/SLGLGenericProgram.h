//#############################################################################
//  File:      SLGLProgramGeneric.h
//  Author:    Marcus Hudritsch
//  Purpose:   Defines a minimal shader program that just starts and stops the
//             shaders that are hold in the base class SLGLProgram.
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLGENERICPROGRAM_H
#define SLGLGENERICPROGRAM_H

#include <SLGLProgram.h>
#include <SLGLProgramManager.h>
#include <SLObject.h>

class SLMaterial;
class SLAssetManager;

//-----------------------------------------------------------------------------
//! Generic Shader Program class inherited from SLGLProgram
/*!
This class only provides the shader begin and end methods. It can be used for
simple GLSL shader programs with standard types of uniform variables.
*/
class SLGLGenericProgram : public SLGLProgram
{
public:
    ~SLGLGenericProgram() override = default;

    //! If s is not NULL, ownership of SLGLProgram is given to SLScene (automatic deletion)
    SLGLGenericProgram(SLAssetManager* s,
                       const SLstring& vertShaderFile,
                       const SLstring& fragShaderFile)
      : SLGLProgram(s, vertShaderFile, fragShaderFile) { ; }

    //! If s is not NULL, ownership of SLGLProgram is given to SLScene (automatic deletion)
    SLGLGenericProgram(SLAssetManager* s,
                       const SLstring& vertShaderFile,
                       const SLstring& fragShaderFile,
                       const SLstring& geomShaderFile)
      : SLGLProgram(s, vertShaderFile, fragShaderFile, geomShaderFile) { ; }

    void beginShader(SLCamera* cam, SLMaterial* mat, SLVLight* lights) override { beginUse(cam, mat, lights); }
    void endShader() override { endUse(); }
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per vertex color shading
class SLGLDefaultProgColorAttrib : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgColorAttrib* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgColorAttrib();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgColorAttrib()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "ColorAttribute.vert",
                           SLGLProgramManager::shaderDir + "Color.frag"){};

    static SLGLDefaultProgColorAttrib* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per vertex lighting without textures
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerVrtBlinn : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerVrtBlinn* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerVrtBlinn();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerVrtBlinn()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerVrtBlinn.vert",
                           SLGLProgramManager::shaderDir + "PerVrtBlinn.frag"){};

    static SLGLDefaultProgPerVrtBlinn* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per vertex lighting with textures
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerVrtBlinnTex : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerVrtBlinnTex* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerVrtBlinnTex();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerVrtBlinnTex()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerVrtBlinnTex.vert",
                           SLGLProgramManager::shaderDir + "PerVrtBlinnTex.frag"){};

    static SLGLDefaultProgPerVrtBlinnTex* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per pixel lighting without textures
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinn : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerPixBlinn* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinn();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerPixBlinn()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinn.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinn.frag"){};

    static SLGLDefaultProgPerPixBlinn* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per pixel lighting with textures mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTex : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerPixBlinnTex* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTex();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerPixBlinnTex()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTex.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTex.frag"){};

    static SLGLDefaultProgPerPixBlinnTex* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per pixel lighting with shadow mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnSM : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerPixBlinnSM* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnSM();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerPixBlinnSM()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnSM.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnSM.frag"){};

    static SLGLDefaultProgPerPixBlinnSM* _instance;
};

//-----------------------------------------------------------------------------
// ! Global default shader program with per pixel lighting with shadow mapping and ambient occlusion
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnSMAO : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerPixBlinnSMAO* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnSMAO();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerPixBlinnSMAO()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnSMAO.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnSMAO.frag"){};

    static SLGLDefaultProgPerPixBlinnSMAO* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per pixel lighting with texture and shadow mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTexSM : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerPixBlinnTexSM* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTexSM();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerPixBlinnTexSM()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexSM.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexSM.frag"){};

    static SLGLDefaultProgPerPixBlinnTexSM* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per pixel lighting with textures and ambient occlusion mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTexAO : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerPixBlinnTexAO* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTexAO();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerPixBlinnTexAO()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexAO.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexAO.frag"){};

    static SLGLDefaultProgPerPixBlinnTexAO* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader with per pixel lighting with texture and normal mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTexNrm : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerPixBlinnTexNrm* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTexNrm();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerPixBlinnTexNrm()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexNrm.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexNrm.frag"){};

    static SLGLDefaultProgPerPixBlinnTexNrm* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader with per pixel lighting with texture, normal and ambient occlusion
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTexNrmAO : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerPixBlinnTexNrmAO* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTexNrmAO();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerPixBlinnTexNrmAO()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexNrmAO.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexNrmAO.frag"){};

    static SLGLDefaultProgPerPixBlinnTexNrmAO* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader with per pixel lighting with texture, normal and shadow mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTexNrmSM : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerPixBlinnTexNrmSM* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTexNrmSM();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerPixBlinnTexNrmSM()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexNrmSM.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexNrmSM.frag"){};

    static SLGLDefaultProgPerPixBlinnTexNrmSM* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader with per pixel lighting with texture, normal, ambient occlusion and shadow mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTexNrmAOSM : public SLGLGenericProgram
{
public:
    static SLGLDefaultProgPerPixBlinnTexNrmAOSM* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTexNrmAOSM();
        return _instance;
    }
    static void deleteInstance()
    {
        if (_instance)
        {
            delete _instance;
            _instance = nullptr;
        }
    }
    static bool isBuilt() { return _instance == nullptr ? false : true; }

private:
    SLGLDefaultProgPerPixBlinnTexNrmAOSM()
      : SLGLGenericProgram(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexNrmAOSM.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTexNrmAOSM.frag"){};

    static SLGLDefaultProgPerPixBlinnTexNrmAOSM* _instance;
};
//-----------------------------------------------------------------------------

#endif
