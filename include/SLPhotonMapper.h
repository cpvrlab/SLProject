//#############################################################################
//  File:      SLPhotonMapper.h
//  Author:    Marcus Hudritsch
//  Date:      September 2011 (HS11)
//  Copyright: M. Hudritsch, Fachhochschule Nordwestschweiz
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPHOTONMAPPER_H
#define SLPHOTONMAPPER_H

#include <stdafx.h>
#include <randomc.h>       // high qualtiy random generators
#include <SLRaytracer.h>
#include <SLPhotonMap.h>

class SLLight;

//-----------------------------------------------------------------------------
typedef enum 
{  LIGHT=0, 
   GLOBAL=1, 
   CAUSTIC=2
} SLPhotonType;
//-----------------------------------------------------------------------------
//! 
/*!      

*/
class SLPhotonMapper: public SLRaytracer
{  public:           
                           SLPhotonMapper ();
                          ~SLPhotonMapper ();
            
            // classic ray tracer functions
            SLbool         render         ();
            SLCol4f        trace          (SLRay* ray);
            SLCol4f        shade          (SLRay* ray);

            void           setPhotonmaps  (SLlong  photonsToEmit,
                                           SLlong  maxCausticStoredPhotons, 
                                           SLuint  maxCausticEstimationPhotons, 
                                           SLfloat maxCausticEstimationRadius,
                                           SLlong  maxGlobalStoredPhotons, 
                                           SLuint  maxGlobalEstimationPhotons, 
                                           SLfloat maxGlobalEstimationRadius);
            void           photonScatter  (SLRay* photon, 
                                           SLVec3f power, 
                                           SLPhotonType photonType);
            void           photonEmission (SLLight* light);

            // Getters
            SLPhotonMap*   mapCaustic        (){return _mapCaustic;}
            SLPhotonMap*   mapGlobal         (){return _mapGlobal;}
            SLlong         photonsToEmit     (){return _photonsToEmit;}
            SLfloat        random            (){return (SLfloat)_random->Random();}
            SLbool         mapCausticGotFull (){return _mapCausticGotFull;}
            SLbool         mapGlobalGotFull  (){return _mapGlobalGotFull;}
            void           mapCausticGotFull (SLbool mapCausticGotFull){_mapCausticGotFull=mapCausticGotFull;}
            void           mapGlobalGotFull  (SLbool mapGlobalGotFull){_mapGlobalGotFull=mapGlobalGotFull;}
                        
   protected:
            // random variables
            TRanrotBGenerator* _random;
            TRanrotBGenerator* _russianRandom;

            // variables for photonmapping
            SLPhotonMap*   _mapCaustic;         //!< pointer to caustics photonmap
            SLPhotonMap*   _mapGlobal;          //!< pointer to global photonmap
            SLbool         _mapCausticGotFull;  //!< holds if changed from not full to full
            SLbool         _mapGlobalGotFull;   //!< holds if changed from not full to full
            SLlong         _photonsToEmit;      //!< total number of photons to be emitted
            SLfloat        _gamma;              //!< gamma correction
};
//-----------------------------------------------------------------------------
#endif
