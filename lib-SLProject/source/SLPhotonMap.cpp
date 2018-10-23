//#############################################################################
//  File:      SLPhotonMap.h
// Authors:    Michael Strub, Stefan Traud
// Date:       25-NOV-2003
//#############################################################################

#include <stdafx.h>
#include "SLPhotonMap.h"

#define swap(ph,a,b) {SLPhoton* ph2=(*ph)[a]; (*ph)[a]=(*ph)[b]; (*ph)[b]=ph2;}

//---------------------------------------------------------------------------
SLPhotonMap::SLPhotonMap()
{
   _storedPhotons=0;
   _lastScaledPhoton=0;
   _maxStoredPhotons=0;
   _maxEstimationPhotons=0;
   _maxEstimationRadius=0;
   _photonList=0;
   _maxPower=SLVec3f(0.0,0.0,0.0);
   _bboxMax=SLVec3f(-SL_FLOAT_MAX,-SL_FLOAT_MAX,-SL_FLOAT_MAX);
   _bboxMin=SLVec3f(SL_FLOAT_MAX,SL_FLOAT_MAX,SL_FLOAT_MAX);
}
//---------------------------------------------------------------------------
SLPhotonMap::~SLPhotonMap()
{
   if (_photonList) delete _photonList;
}
//---------------------------------------------------------------------------
/*!
Stores the photons into a unsorted photon-list
*/
void SLPhotonMap::store(SLVec3f pos, SLVec3f dir, SLVec3f power)
{
   if (_storedPhotons>=_maxStoredPhotons)
      return;
   
   _storedPhotons++;
   (*_photonList)[_storedPhotons].pos.x=(SLfloat)pos.x;
   (*_photonList)[_storedPhotons].pos.y=(SLfloat)pos.y;
   (*_photonList)[_storedPhotons].pos.z=(SLfloat)pos.z;

   (*_photonList)[_storedPhotons].dir.x=(SLfloat)dir.x;
   (*_photonList)[_storedPhotons].dir.y=(SLfloat)dir.y;
   (*_photonList)[_storedPhotons].dir.z=(SLfloat)dir.z;

   (*_photonList)[_storedPhotons].power=RGBE(power);

   for (int i=0; i<3; i++)
   {
      if(pos.comp[i] > _bboxMax.comp[i]) _bboxMax.comp[i] = pos.comp[i];
      if(pos.comp[i] < _bboxMin.comp[i]) _bboxMin.comp[i] = pos.comp[i];
   }
   
}
//---------------------------------------------------------------------------
/*!
Scales the photon-power after each lightsource by the amount of photons
emitted by this lightsource
*/
void SLPhotonMap::scalePhotonPower(SLfloat scale)
{
   for (SLlong i=(_lastScaledPhoton);i<=_storedPhotons;++i)
   {
      SLVec3f power=(*_photonList)[i].power.getRGBE() * scale;
      (*_photonList)[i].power=RGBE(power);

      ////////PHOTONMAP PREVIE PARAMS/////////////////////////////////////
      // sets params used for scaling the photons for the photonmap-preview
      
      if(power.x > _maxPower.x) _maxPower.x = power.x;
      if(power.y > _maxPower.y) _maxPower.y = power.y;
      if(power.z > _maxPower.z) _maxPower.z = power.z;
      
      _sumPower += power;
      //
      ////////////////////////////////////////////////////////////////////

   }
   _lastScaledPhoton = _storedPhotons+1;
}

//---------------------------------------------------------------------------
/*!
Returns the irradiance which is received by the given point
Additionally a filter can be chosen to supress the "clouds" caused by the estimation
*/
SLCol4f SLPhotonMap::irradianceEstimate (SLVec3f pos, SLVec3f normal, SLFilterType filter)
{
   SLVec3f irrad(0.0,0.0,0.0);
   SLCol4f result(0.0,0.0,0.0);

   //if the photonmap is empty the estimation can be aborted
   if (_storedPhotons==0)
      return result;

   ///////NEAREST PHOTONS////////////////////////////////////////
   // initializing the struct used to locate the nearest photons
   //
   SLNearestPhotons np;
   np.dist2    = new SLfloat [_maxEstimationPhotons+1];
   np.index    = new SLPhotonPointerList(_maxEstimationPhotons+1);
   np.pos.x    = (SLfloat) pos.x;
   np.pos.y    = (SLfloat) pos.y;
   np.pos.z    = (SLfloat) pos.z;
   np.max      = _maxEstimationPhotons;
   np.found    = 0;
   np.got_heap = 0;
   np.dist2[0] = _maxEstimationRadius*_maxEstimationRadius;
   //
   ///////////////////////////////////////////////////////////////


   // locate the nearest photons
   locatePhotons(&np,1);


   ///STATISTIC/////////////////////////////////////////////
   //
   if(np.dist2[0]>_maxRadius)    _maxRadius = np.dist2[0];
   if(np.found>_maxPhotonsFound) _maxPhotonsFound = np.found;
   //
   /////////////////////////////////////////////////////////


   // if less than 8 photons return
   if (np.found<8)
   {
      if(np.dist2) delete [] np.dist2;
      if(np.index) delete np.index;
      return result;
   }

   ///////IRRADIANCE//////////////////////////////////////////////////////////////
   // sum irradiance from all photons
   //
   SLVec3f power;
   for (SLlong i=1; i<=np.found; i++)
   {
      const SLPhoton* p = (*(np.index))[i];

      // the following "if" can be omitted if the scene does not have any thin surfaces
      if ( (p->dir.x*(SLfloat)normal.x + p->dir.y*(SLfloat)normal.y + p->dir.z*(SLfloat)normal.z) < 0.0f)
      {
         power=p->power.getRGBE();

         // if a filter is selected the photon-power has to be weighted
         switch (filter)
         {
            case NONE:
               irrad += power;
               break;
            case CONE:
               irrad += power * (1.0f - (sqrt(np.dist2[i])/1.1f/sqrt(np.dist2[0])));
               break;
            case GAUSS:
               irrad += power * (0.918f*(1.0f-(1.0f-exp(-1.953f*np.dist2[i]/(2.0f*np.dist2[0])))/(1.0f-exp(-1.953f))));
               break;
         }
      }
   }
   //
   ///////////////////////////////////////////////////////////////////////////////

   const SLfloat tmp = (1.0f/SL_PI)/np.dist2[0]; //estimate of density

   irrad *= tmp;
   result.set(irrad.x,irrad.y,irrad.z);

   if(np.dist2) delete [] np.dist2;
   if(np.index) delete np.index;
   
   return result;
}

//---------------------------------------------------------------------------
/*!
Locates the n nearest photons to a given point by getting called recursively
If more than n photons are found, rearranges the list of the nearest photons
as a heap for fast deleting the farest photon and inserting a nearer photon
The result is stored in the SLNearestPhotons pointer
*/
void SLPhotonMap::locatePhotons (SLNearestPhotons* const np, const SLlong index) const
{
   SLPhoton* p = &((*_photonList)[index]);
   SLfloat dist1;

   //check if the photon at "index" has any children
   if (index < (_storedPhotons/2.0-1.0))//half_stored_photons
   {
      dist1 = np->pos.comp[SLint(p->plane)] - p->pos.comp[SLint(p->plane)];

      if (dist1>0.0) // if dist1 is positive search right plane
      {
         locatePhotons(np, 2*index+1);
         if (dist1*dist1 < np->dist2[0])
            locatePhotons(np,2*index);
      }
      else // dist1 is negative search left first
      {
         locatePhotons(np, 2*index);
         if (dist1*dist1 < np->dist2[0])
            locatePhotons(np, 2*index+1);
      }
   }

   // compute squared distance between current photon and np->pos
   dist1 = p->pos.x - np->pos.x;
   SLfloat dist2 = dist1*dist1;
   dist1 = p->pos.y - np->pos.y;
   dist2 += dist1*dist1;
   dist1 = p->pos.z - np->pos.z;
   dist2 += dist1*dist1;

   // check if the photon is within the max search radius
   if (dist2 < np->dist2[0])
   {  
      //check if we already found max count of nearest photons
      if (np->found < np->max)
      {  // list is not full, just append the photon to the list
         np->found++;
         np->dist2[np->found] = dist2;
         (*(np->index))[np->found] = p;
      }
      else
      {
         // list is full, photon has to be inserted into a heap (kind of priority queue)
         SLlong j,parent;
         
         //check if the list of nearest photons is already arranged as heap
         if (np->got_heap==0)
         {  
            /////BUILD HEAP////////////////////////////////////////////////
            //
            float dst2;
            SLPhoton* phot;
            SLlong halfFound = np->found>>1;// = np->found/2.0 + 1.0;//
            for (SLlong k=halfFound; k>=1; k--)
            {
               parent=k;
               phot = (*np->index)[k];
               dst2 = np->dist2[k];
               while (parent <= halfFound)
               {
                  j = parent+parent;
                  if (j<np->found && np->dist2[j]<np->dist2[j+1])
                     j++;
                  if (dst2>=np->dist2[j])
                     break;
                  np->dist2[parent] = np->dist2[j];
                  (*np->index)[parent] = (*np->index)[j];
                  parent=j;
               }
               np->dist2[parent] = dst2;
               (*np->index)[parent] = phot;
            }
            np->got_heap = 1;
            //
            ///////////////////////////////////////////////////////////////
         }
         
         ///////REARRANGE HEAP//////////////////////////////////////
         // insert new photon into max heap
         // delete largest element, insert new, and reorder the heap
         //
         parent=1;
         j = 2;
         while (j <= np->found)
         {
            if (j < np->found && np->dist2[j] < np->dist2[j+1])
               j++;
            if (dist2 > np->dist2[j])
               break;
            np->dist2[parent] = np->dist2[j];
            (*np->index)[parent] = (*np->index)[j];
            parent = j;
            j += j;
         }
         (*np->index)[parent] = p;
         np->dist2[parent] = dist2;

         np->dist2[0] = np->dist2[1];
         //
         ///////////////////////////////////////////////////////////
      }
   }
}

//---------------------------------------------------------------------------
/*!
Arranges the unsorted photon-list as a heap-like kd-tree by calling the
recursivly balanceSegment method
*/
void SLPhotonMap::balance ()
{
   if(_storedPhotons>1)
   {
      _balanced=0;
      //allocate two temporary arrays of pointers for the balancing procedure
      SLPhotonPointerList* pa1 = new SLPhotonPointerList(_storedPhotons+1);
      SLPhotonPointerList* pa2 = new SLPhotonPointerList(_storedPhotons+1);

      SLlong i;
      for(i=0;i<=_storedPhotons;i++)
      {
         (*pa2)[i] = &(*_photonList)[i];
      }
      
      // call of the recursive balanceSegment method to sort the list as kd-tree
      balanceSegment(pa1,pa2,1,1,_storedPhotons);
      delete(pa2);

      //////////////////////////////////////////////////////////////////////
      //reorganize balanced kd-tree (make a heap)
      //
      SLlong d, j=1, foo=1;
      SLPhoton fooPhoton = (*_photonList)[j];

      for (i=1;i<=_storedPhotons;i++)
      {
         // calculates the index using the adressdistance
         d = ((SLlong)(*pa1)[j]-(SLlong)&(*_photonList)[0])/sizeof(SLPhoton);
         (*pa1)[j]=NULL;
         if(d != foo)
            (*_photonList)[j] = (*_photonList)[d];
         else
         {
            (*_photonList)[j] = fooPhoton;

            if (i<_storedPhotons)
            {
               for (;foo<=_storedPhotons;foo++)//foo already initialized
               {  if ((*pa1)[foo] != NULL)
                     break;
               }
               fooPhoton = (*_photonList)[foo];
               j=foo;
            }
            continue;
         }
         j=d;
      } 
      delete(pa1);
      //
      //////////////////////////////////////////////////////////////////////
   }
}
//---------------------------------------------------------------------------
/*!
Recursive function to arrange the unsorted photon-list as a heap-like kd-tree
Computes the median of the given segment, finds the axis to split along, and
calls itself recursivly for the new built segments
*/
void SLPhotonMap::balanceSegment (SLPhotonPointerList* pbal,SLPhotonPointerList* porg,const SLlong index,const SLlong start,const SLlong end)
{
   // progress
   SLlong received = end-start+1;

   //////MEDIAN/////////////////////////////////////////
   // compute new median
   // ensures that the kd-tree is left-balanced
   //
   SLlong median=1;
   while ((4*median) <= (end-start+1))
      median += median;

   if ((3*median) <= (end-start+1))
   {
      median += median;
      median += start-1;
   }
   else
      median = end-median+1;
   //
   ////////////////////////////////////////////////////


   ////////////////////////////////////////////////////
   // find axis to split along
   //
   int axis=2;
   if ((_bboxMax.x-_bboxMin.x)>(_bboxMax.y-_bboxMin.y) &&
       (_bboxMax.x-_bboxMin.x)>(_bboxMax.z-_bboxMin.z))
       axis=0;
   else if ((_bboxMax.y-_bboxMin.y)>(_bboxMax.z-_bboxMin.z))
      axis=1;
   //
   ////////////////////////////////////////////////////


   ////////////////////////////////////////////////////
   // partition photon block around the median
   //
   medianSplit(porg,start,end,median,axis);

   (*pbal)[index] = (*porg)[median];
   (*pbal)[index]->plane = axis;
   //
   ////////////////////////////////////////////////////

   if (received>1000&&_balanced>0) printf("\b\b\b\b%3.0f%%",(float)_balanced/_storedPhotons*100.0);
   ////////////////////////////////////////////////////
   // recursively balance the left and right block
   //
   if (median > start)
   {
      // balance left segment
      if (start < median-1)
      {
         const SLfloat tmp=_bboxMax.comp[axis];
         _bboxMax.comp[axis] = (*pbal)[index]->pos.comp[axis];
         balanceSegment(pbal, porg, 2*index, start, median-1);
         _bboxMax.comp[axis] = tmp;

         //progress
         received-=median-start;
      }
      else
         (*pbal)[2*index] = (*porg)[start];
   }

   if (median < end)
   {
      // balance right segment
      if (median+1 < end)
      {
         const SLfloat tmp = _bboxMin.comp[axis];
         _bboxMin.comp[axis] = (*pbal)[index]->pos.comp[axis];
         balanceSegment(pbal, porg, 2*index+1, median+1, end);
         _bboxMin.comp[axis] = tmp;

         //progress
         received-= end - median;
      }
      else
         (*pbal)[2*index+1] = (*porg)[end];
   }
   //
   ////////////////////////////////////////////////////

      //progress
      _balanced += received;
}

//---------------------------------------------------------------------------
/*!
Rearranges the photons of a given segment to assure that photons whose coordinate
for the given axis are smaller than the one of the median are on the left side
and the others are on the right side of the mdeian
*/
void SLPhotonMap::medianSplit (SLPhotonPointerList* p, const SLlong start, const SLlong end, const SLlong median, const SLint  axis)
{
   SLlong left = start;
   SLlong right = end;
   
   while (right > left)
   {  
      //PROGRESS////////////////////////////////////////
      //
      if(_balanced<1000)
      {
         _char++;
         switch(_char)
         {
            case 0:
            case 4: printf("\b%c",'/'); break;
            case 1:
            case 5: printf("\b%c",'-'); break;
            case 2:
            case 6: printf("\b%c",'\\');break;
            case 3: printf("\b%c",'|'); break;
            case 7: printf("\b%c",'|'); _char=-1; break;
         }
      }
      //
      //////////////////////////////////////////////////

      const SLfloat v = (*p)[right]->pos.comp[axis];
      SLlong i = left-1;
      SLlong j = right;
      for(;;)//while(true)
      {
         while ((*p)[++i]->pos.comp[axis] < v)
            ;
         while ((*p)[--j]->pos.comp[axis] > v && j>left)
            ;
         if(i>=j)
            break;//exit the endless loop
         swap(p,i,j)
      }
      swap(p,i,right);
      if (i >= median)
         right=i-1;
      if (i <= median)
         left = i+1;
   }
}

//---------------------------------------------------------------------------
/*!
Stores the maximum photons to be stored, the max count of photons to be located
during the radiance estimate and the max search radius
*/
void SLPhotonMap::setPhotonMapParams(SLlong maxStoredPhotons, 
                                     SLuint maxEstimationPhotons, 
                                     SLfloat maxEstimationRadius)
{
   _maxStoredPhotons=maxStoredPhotons;
   _maxEstimationPhotons=maxEstimationPhotons;
   _maxEstimationRadius=maxEstimationRadius;
   _storedPhotons=0;
   _lastScaledPhoton=0;

   if (_photonList) delete _photonList;
   _photonList = new SLPhotonList(_maxStoredPhotons+1);
}
//---------------------------------------------------------------------------
void SLPhotonMap::setTestParams(SLuint maxEstimationPhotons,
                                SLfloat maxEstimationRadius)
{
   std::cout<<maxEstimationPhotons<<std::endl;
   std::cout<<maxEstimationRadius<<std::endl;
   _maxEstimationPhotons=maxEstimationPhotons;
   _maxEstimationRadius=maxEstimationRadius;
}
//-----------------------------------------------------------------------------
