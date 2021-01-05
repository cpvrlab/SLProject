//#############################################################################
//  File:      SLRay.h
//  Author:    Michael Strub, Stefan Traud
//  Date:      September 2011 (HS11)
//  Copyright: Michael Strub, Stefan Traud, Fachhochschule Nordwestschweiz
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPHOTONMAP_H
#define SLPHOTONMAP_H

#include <stdafx.h>
#include <renderbitch/rgbe.h>

//-----------------------------------------------------------------------------
// photon struct
typedef struct 
{  SLVec3f  pos;     // position of the photon
   SLVec3f  dir;     // origin direction
   SLbyte   plane;   // splitting plane
   RGBE     power;   // power compressed as Ward's Shared Exponent RGB-format
} SLPhoton;
typedef std::vector<SLPhoton>  SLPhotonList;
typedef std::vector<SLPhoton*> SLPhotonPointerList;

//-----------------------------------------------------------------------------
// nearest photons struct
typedef struct 
{  SLlong   max;        // max amount of photons to locate
   SLlong   found;      // amount of found photons
   SLbool   got_heap;   // bool indicating if nearest photon are arranged as heap
   SLVec3f  pos;        // position of the point P in the scene
   SLfloat* dist2;      // array holding the distance of the photons to point P
   SLPhotonPointerList* index; //! array of pointers to the photons in the photonmap
} SLNearestPhotons;

//-----------------------------------------------------------------------------
typedef enum 
{  NONE=0, 
   CONE=1, 
   GAUSS=2
} SLFilterType;

//-----------------------------------------------------------------------------
//! Photonmap class with datastructure, scattering and gattering methods
/*!
Holds a datastructure for storing photons and diffrent methods for the photon
scattering and the radiance estimate. The datastructure is a heaplike left-
balanced kd-tree. It is implented following the source code from H. W. Jensen
described in his book "Realistic Image Synthesis Using Photon Mappin.
*/
class SLPhotonMap
{
   public:
      SLPhotonMap();
      ~SLPhotonMap();

      //Scattering methods
      void     store                (SLVec3f pos, SLVec3f dir, SLVec3f power);
      void     scalePhotonPower     (SLfloat scale);

      //Gattering methods
      SLCol4f  irradianceEstimate   (SLVec3f pos, SLVec3f normal, SLFilterType filterType=NONE);
      void     locatePhotons        (SLNearestPhotons* const np, const SLlong index) const;

      //Balancing methods
      void     balance              ();
      void     balanceSegment       (SLPhotonPointerList* pbal,
                                     SLPhotonPointerList* porg,
                                     const SLlong index,
                                     const SLlong start,
                                     const SLlong end
                                    );
      void     medianSplit          (SLPhotonPointerList* p,
                                     const SLlong start,
                                     const SLlong end,
                                     const SLlong median,
                                     const SLint  axis
                                    );

      //additional methods
      void     setPhotonMapParams   (SLlong maxStoredPhotons,
                                     SLuint maxEstimationPhotons,
                                     SLfloat maxEstimationRadius
                                    );
      void     setTestParams        (SLuint maxEstimationPhotons,
                                     SLfloat maxEstimationRadius);
      SLbool   isDone               ()             {return _storedPhotons>0;};
      SLbool   isFull               ()             {return _storedPhotons>=_maxStoredPhotons;}
      SLlong   storedPhotons        ()             {return _storedPhotons;}
      SLVec3f  photonPosition       (SLlong index) {SLVec3f pos((*_photonList)[index+1].pos);
                                                    return SLVec3f((SLfloat)pos.x,(SLfloat) pos.y,(SLfloat)pos.z); }
      SLVec3f  photonPower          (SLlong index) {return (*_photonList)[index+1].power.getRGBE();}
      SLVec3f  maxPower             ()             {return _maxPower;};
      SLVec3f  avgPower             ()             {return _sumPower/(SLfloat)_storedPhotons;};
      SLfloat  maxRadius            ()             {return sqrt(_maxRadius);};
      SLlong   maxPhotonsFound      ()             {return _maxPhotonsFound;};
      SLlong   maxStoredPhotons     ()             {return _maxStoredPhotons;};
      void     clearMax             ()             {_maxRadius=0;_maxPhotonsFound=0;};

   private:
      SLPhotonList* _photonList;       //!< the photonmap structure
      SLlong   _storedPhotons;         //!< amount of stored photons
      SLlong   _lastScaledPhoton;      //!< stores the index of the last scaled photon
      SLlong   _maxStoredPhotons;      //!< max amount of photons to store in the photonmap
      SLuint   _maxEstimationPhotons;  //!< max amount of nearest photons to locate
      SLfloat  _maxEstimationRadius;   //!< max search radius to locate the nearest photons
      SLVec3f  _bboxMax;               //!< max values for x,y,z coordinates
      SLVec3f  _bboxMin;               //!< min values for x,y,z coordinates
      SLVec3f  _maxPower;              //!< max stored power (used for drawing the photonmap in OGL)
      SLVec3f  _sumPower;              //!< (used for drawing the photonmap in OGL)
      SLfloat  _maxRadius;             //!< (used for statistic output concerning the estimation)
      SLlong   _maxPhotonsFound;       //!< (used for statistic output concerning the estimation)
      SLlong   _balanced;              //!< (used for indicating progress of balancing)
      SLlong   _char;                  //!< (used for indicating progress of balancing)
};
//-----------------------------------------------------------------------------
#endif