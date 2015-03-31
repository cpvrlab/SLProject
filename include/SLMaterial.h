//#############################################################################
//  File:      SLMaterial.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLMAT_H
#define SLMAT_H

#include <stdafx.h>
#include <SLDrawBits.h>
#include <SLGLTexture.h>
#include <SLGLProgram.h>
#include <SLGLGenericProgram.h>

class SLSceneView;
class SLNode;

//-----------------------------------------------------------------------------
//! Defines a standard CG material with textures and a shader program
/*!      
The SLMatrial class defines a material with properties for ambient, diffuse, 
specular and emissive light RGBA-reflection. In addition it has coeffitients
for reflectivity (kr), transparency (kt) and refraction index (kn) that can be
used in special shaders and ray tracing.
*/
class SLMaterial : public SLObject
{  public:
                            //! Default ctor
                            SLMaterial(const SLchar* name=(SLchar*)"Default material",
                                       SLCol4f amdi=SLCol4f::WHITE, 
                                       SLCol4f spec=SLCol4f::WHITE,
                                       SLfloat shininess=100.0f,
                                       SLfloat kr=0.0, 
                                       SLfloat kt=0.0f, 
                                       SLfloat kn=1.0f);
                           
                            //! Ctor for textures
                            SLMaterial(const SLchar* name,
                                       SLGLTexture* texture1,
                                       SLGLTexture* texture2=0,
                                       SLGLTexture* texture3=0,
                                       SLGLTexture* texture4=0,
                                       SLGLProgram* program=0);

                            //! Ctor for uniform color material without lighting
                            SLMaterial(SLCol4f uniformColor, 
                                       const SLchar* name=(SLchar*)"Uniform color");
                           
                            //! Copy ctor
                            SLMaterial(SLMaterial* m) {if (m) set(m);}
                           
                           ~SLMaterial();
            
            //! Sets the material states and passes all variables to the shader program 
            void            activate       (SLGLState* state,
                                            SLDrawBits drawBits);
           
            //! Returns true if there is any transparency in diffuse alpha or textures
            SLbool          hasAlpha()        {return (_diffuse.a < 1.0f || 
                                                       (_textures.size() && 
                                                        _textures[0]->hasAlpha()));}
            
            //! Returns true if a material has a 3D texture
            #ifdef SL_GLES2
            SLbool          has3DTexture   () {return false;}
            #else
            SLbool          has3DTexture   () {return _textures.size() > 0 &&
                                                      _textures[0]->target()==GL_TEXTURE_3D;}
            #endif
            //! Returns true if a material with textures tangents as additional attributes
            SLbool          needsTangents  () {return (_textures.size()>=2 &&
                                                       _textures[0]->target()==GL_TEXTURE_2D &&
                                                       _textures[1]->texType()==NormalMap);}
            // Setters
            void            set             (SLMaterial* m);
            void            ambient         (SLCol4f ambi)    {_ambient = ambi;}
            void            diffuse         (SLCol4f diff)    {_diffuse = diff;}
            void            ambientDiffuse  (SLCol4f am_di)   {_ambient = _diffuse = am_di;}
            void            specular        (SLCol4f spec)    {_specular = spec;}
            void            emission        (SLCol4f emis)    {_emission = emis;}
            void            transmission    (SLCol4f transm)  {_transmission = transm;}
            void            translucency    (SLfloat transl)  {_translucency = transl;}
            void            shininess       (SLfloat shin)    {if(shin<0.0f) shin=0.0;
                                                              _shininess = shin;}
            void            kr              (SLfloat kr)      {if(kr<0.0f) kr=0.0f;
                                                               if(kr>1.0f) kr=1.0f;
                                                              _kr = kr;}
            void            kt              (SLfloat kt)      {if(kt<0.0f) kt=0.0f;
                                                               if(kt>1.0f) kt=1.0f;
                                                              _kt = kt;
                                                              _ambient.w  = 1.0f-kt;
                                                              _diffuse.w  = 1.0f-kt;
                                                              _specular.w = 1.0f-kt;}
            void            kn              (SLfloat kn)      {assert(kn>=0.0f);
                                                              _kn = kn;}  
            void            program      (SLGLProgram* sp){_program = sp;}
            
            // Getters
            SLCol4f         ambient         () {return _ambient;}
            SLCol4f         diffuse         () {return _diffuse;}
            SLCol4f         specular        () {return _specular;}
            SLCol4f         transmission    () {return _transmission;}
            SLCol4f         emission        () {return _emission;}
            SLfloat         shininess       () {return _shininess;}
            SLfloat         translucency    () {return _translucency;}
            SLfloat         kr              () {return _kr;}
            SLfloat         kt              () {return _kt;}
            SLfloat         kn              () {return _kn;}
            SLVGLTexture&   textures        () {return _textures;}
            SLGLProgram*    program         () {return _program;}

     static SLMaterial*     defaultMaterial ();
     static void            defaultMaterial (SLMaterial* mat);


     // Static variables & functions
     static SLMaterial*     current;        //!< Current material during scene traversal
     static SLfloat         K;              //!< PM: Constant of gloss calibration (slope of point light at dist 1)
     static SLfloat         PERFECT;        //!< PM: shininess/translucency limit

   protected:    
            SLCol4f         _ambient;       //!< ambient color (RGB reflection coefficients)
            SLCol4f         _diffuse;       //!< diffuse color (RGB reflection coefficients)
            SLCol4f         _specular;      //!< specular color (RGB reflection coefficients)
            SLCol4f         _transmission;  //!< PM: transmissive color (RGB reflection coefficients)
            SLCol4f         _emission;      //!< emissive color coefficients
            SLfloat         _shininess;     //!< shininess exponent for light reflections
            SLfloat         _translucency;  //!< PM: translucency exponent for light refraction
            SLfloat         _kr;            //!< reflection coefficient 0.0 - 1.0
            SLfloat         _kt;            //!< transmission coefficient 0.0 - 1.0
            SLfloat         _kn;            //!< refraction index
            SLVGLTexture    _textures;      //!< vector of texture pointers
            SLGLProgram*    _program;       //!< pointer to a GLSL shader program

    private:
    static  SLMaterial*     _defaultMaterial;//!< Global default material for meshes that don't define their own.
};
//-----------------------------------------------------------------------------
//! STL vector of material pointers
typedef std::vector<SLMaterial*>  SLVMaterial;
//-----------------------------------------------------------------------------
#endif
