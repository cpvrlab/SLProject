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

#ifndef SLGLPROGRAMMGENERIC_H
#define SLGLPROGRAMMGENERIC_H

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
class SLGLProgramGeneric : public SLGLProgram
{
public:
    ~SLGLProgramGeneric() override = default;

    //! If s is not NULL, ownership of SLGLProgram is given to SLScene (automatic deletion)
    SLGLProgramGeneric(SLAssetManager* am,
                       const SLstring& vertShaderFile,
                       const SLstring& fragShaderFile)
      : SLGLProgram(am, vertShaderFile, fragShaderFile) { ; }

    //! If s is not NULL, ownership of SLGLProgram is given to SLScene (automatic deletion)
    SLGLProgramGeneric(SLAssetManager* am,
                       const SLstring& vertShaderFile,
                       const SLstring& fragShaderFile,
                       const SLstring& geomShaderFile)
      : SLGLProgram(am, vertShaderFile, fragShaderFile, geomShaderFile) { ; }

    void beginShader(SLCamera* cam, SLMaterial* mat, SLVLight* lights) override { beginUse(cam, mat, lights); }
    void endShader() override { endUse(); }
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per vertex color shading
class SLGLDefaultProgColorAttrib : public SLGLProgramGeneric
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgColorAttrib()
      : SLGLProgramGeneric(nullptr,
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
class SLGLDefaultProgPerVrtBlinn : public SLGLProgramGeneric
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerVrtBlinn()
      : SLGLProgramGeneric(nullptr,
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
class SLGLDefaultProgPerVrtBlinnTm : public SLGLProgramGeneric
{
public:
    static SLGLDefaultProgPerVrtBlinnTm* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerVrtBlinnTm();
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerVrtBlinnTm()
      : SLGLProgramGeneric(nullptr,
                           SLGLProgramManager::shaderDir + "PerVrtBlinnTm.vert",
                           SLGLProgramManager::shaderDir + "PerVrtBlinnTm.frag"){};

    static SLGLDefaultProgPerVrtBlinnTm* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per pixel lighting without textures
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinn : public SLGLProgramGeneric
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerPixBlinn()
      : SLGLProgramGeneric(nullptr,
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
class SLGLDefaultProgPerPixBlinnTex : public SLGLProgramGeneric
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerPixBlinnTex()
      : SLGLProgramGeneric(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTm.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTm.frag"){};

    static SLGLDefaultProgPerPixBlinnTex* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per pixel lighting with shadow mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnSm : public SLGLProgramGeneric
{
public:
    static SLGLDefaultProgPerPixBlinnSm* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnSm();
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerPixBlinnSm()
      : SLGLProgramGeneric(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnSm.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnSm.frag"){};

    static SLGLDefaultProgPerPixBlinnSm* _instance;
};

//-----------------------------------------------------------------------------
// ! Global default shader program with per pixel lighting with shadow mapping and ambient occlusion
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnAoSm : public SLGLProgramGeneric
{
public:
    static SLGLDefaultProgPerPixBlinnAoSm* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnAoSm();
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerPixBlinnAoSm()
      : SLGLProgramGeneric(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnAoSm.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnAoSm.frag"){};

    static SLGLDefaultProgPerPixBlinnAoSm* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per pixel lighting with texture and shadow mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTmSm : public SLGLProgramGeneric
{
public:
    static SLGLDefaultProgPerPixBlinnTmSm* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTmSm();
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerPixBlinnTmSm()
      : SLGLProgramGeneric(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmSm.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmSm.frag"){};

    static SLGLDefaultProgPerPixBlinnTmSm* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader program with per pixel lighting with textures and ambient occlusion mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTexAO : public SLGLProgramGeneric
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerPixBlinnTexAO()
      : SLGLProgramGeneric(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmAo.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmAo.frag"){};

    static SLGLDefaultProgPerPixBlinnTexAO* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader with per pixel lighting with texture and normal mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTmNm : public SLGLProgramGeneric
{
public:
    static SLGLDefaultProgPerPixBlinnTmNm* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTmNm();
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerPixBlinnTmNm()
      : SLGLProgramGeneric(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmNm.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmNm.frag"){};

    static SLGLDefaultProgPerPixBlinnTmNm* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader with per pixel lighting with texture, normal and ambient occlusion
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTmNmAo : public SLGLProgramGeneric
{
public:
    static SLGLDefaultProgPerPixBlinnTmNmAo* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTmNmAo();
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerPixBlinnTmNmAo()
      : SLGLProgramGeneric(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmNmAo.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmNmAo.frag"){};

    static SLGLDefaultProgPerPixBlinnTmNmAo* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader with per pixel lighting with texture, normal and shadow mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTmNmSm : public SLGLProgramGeneric
{
public:
    static SLGLDefaultProgPerPixBlinnTmNmSm* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTmNmSm();
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerPixBlinnTmNmSm()
      : SLGLProgramGeneric(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmNmSm.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmNmSm.frag"){};

    static SLGLDefaultProgPerPixBlinnTmNmSm* _instance;
};
//-----------------------------------------------------------------------------
// ! Global default shader with per pixel lighting with texture, normal, ambient occlusion and shadow mapping
/*!
 * This default shader program is dependant from the number of lights in a
 * scene and must therefore be deallocated at scene destruction.
 */
class SLGLDefaultProgPerPixBlinnTmNmAoSm : public SLGLProgramGeneric
{
public:
    static SLGLDefaultProgPerPixBlinnTmNmAoSm* instance()
    {
        if (!_instance)
            _instance = new SLGLDefaultProgPerPixBlinnTmNmAoSm();
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
    static bool isBuilt() { return _instance != nullptr; }

private:
    SLGLDefaultProgPerPixBlinnTmNmAoSm()
      : SLGLProgramGeneric(nullptr,
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmNmAoSm.vert",
                           SLGLProgramManager::shaderDir + "PerPixBlinnTmNmAoSm.frag"){};

    static SLGLDefaultProgPerPixBlinnTmNmAoSm* _instance;
};
//-----------------------------------------------------------------------------

#endif
