/**** EulerAngles.c - Convert Euler angles to/from matrix or quat ****/
/* Ken Shoemake, 1993 */
#include <math.h>
#include <float.h>
#include "EulerAngles.h"

EulerAngles Eul_(float ai, float aj, float ah, int order)
{
    EulerAngles ea;
    ea.x = ai; ea.y = aj; ea.z = ah;
    ea.w = (float)order;
    return (ea);
}
/* Construct quaternion from Euler angles (in radians). */
#pragma warning(disable:4552)
#pragma warning( disable:4244)
Quat Eul_ToQuat(EulerAngles ea)
{
    Quat qu;
    double a[3], ti, tj, th, ci, cj, ch, si, sj, sh, cc, cs, sc, ss;
    int i,j,k,h,n,s,f;
    EulGetOrd(ea.w,i,j,k,h,n,s,f);
    if (f==EulFrmR) {float t = ea.x; ea.x = ea.z; ea.z = t;}
    if (n==EulParOdd) ea.y = -ea.y;
    ti = ea.x*0.5; tj = ea.y*0.5; th = ea.z*0.5;
    ci = cos(ti);  cj = cos(tj);  ch = cos(th);
    si = sin(ti);  sj = sin(tj);  sh = sin(th);
    cc = ci*ch; cs = ci*sh; sc = si*ch; ss = si*sh;
    if (s==EulRepYes) {
   a[i] = cj*(cs + sc);	/* Could speed up with */
   a[j] = sj*(cc + ss);	/* trig identities. */
   a[k] = sj*(cs - sc);
   qu.w = cj*(cc - ss);
    } else {
   a[i] = cj*sc - sj*cs;
   a[j] = cj*ss + sj*cc;
   a[k] = cj*cs - sj*sc;
   qu.w = cj*cc + sj*ss;
    }
    if (n==EulParOdd) a[j] = -a[j];
    qu.x = a[X]; qu.y = a[Y]; qu.z = a[Z];
    return (qu);
}

/* Construct matrix from Euler angles (in radians). */
#pragma warning(disable:4552)
#pragma warning( disable:4244)
void Eul_ToHMatrix(EulerAngles ea, HMatrix M)
{  double ti, tj, th, ci, cj, ch, si, sj, sh, cc, cs, sc, ss;
   int i,j,k,h,n,s,f;
   EulGetOrd(ea.w,i,j,k,h,n,s,f);
   if (f==EulFrmR) {float t = ea.x; ea.x = ea.z; ea.z = t;}
   if (n==EulParOdd) {ea.x = -ea.x; ea.y = -ea.y; ea.z = -ea.z;}
   ti = ea.x;	  tj = ea.y;	th = ea.z;
   ci = cos(ti); cj = cos(tj); ch = cos(th);
   si = sin(ti); sj = sin(tj); sh = sin(th);
   cc = ci*ch; cs = ci*sh; sc = si*ch; ss = si*sh;
   if (s==EulRepYes) 
   {  M[i][i]=(float)(cj);	    M[i][j]=(float)( sj*si);    M[i][k]=(float)( sj*ci);
      M[j][i]=(float)(sj*sh);  M[j][j]=(float)(-cj*ss+cc); M[j][k]=(float)(-cj*cs-sc);
      M[k][i]=(float)(-sj*ch); M[k][j]=(float)( cj*sc+cs); M[k][k]=(float)( cj*cc-ss);
   } else 
   {  M[i][i]=(float)(cj*ch);  M[i][j]=(float)( sj*sc-cs); M[i][k]=(float)(sj*cc+ss);
      M[j][i]=(float)(cj*sh);  M[j][j]=(float)( sj*ss+cc); M[j][k]=(float)(sj*cs-sc);
      M[k][i]=(float)(-sj);	 M[k][j]=(float)( cj*si);    M[k][k]=(float)(cj*ci);
   }
   M[W][X]=M[W][Y]=M[W][Z]=M[X][W]=M[Y][W]=M[Z][W]=0.0f; M[W][W]=1.0f;
}

/* Convert matrix to Euler angles (in radians). */
#pragma warning(disable:4552)
#pragma warning( disable:4244)
EulerAngles Eul_FromHMatrix(HMatrix M, int order)
{
    EulerAngles ea;
    int i,j,k,h,n,s,f;
    EulGetOrd(order,i,j,k,h,n,s,f);
    if (s==EulRepYes) {
   double sy = sqrt(M[i][j]*M[i][j] + M[i][k]*M[i][k]);
   if (sy > 16*FLT_EPSILON) {
       ea.x = atan2(M[i][j], M[i][k]);
       ea.y = atan2((float)sy, M[i][i]);
       ea.z = atan2(M[j][i], -M[k][i]);
   } else {
       ea.x = atan2(-M[j][k], M[j][j]);
       ea.y = atan2((float)sy, M[i][i]);
       ea.z = 0;
   }
    } else {
   double cy = sqrt(M[i][i]*M[i][i] + M[j][i]*M[j][i]);
   if (cy > 16*FLT_EPSILON) {
       ea.x = atan2(M[k][j], M[k][k]);
       ea.y = atan2(-M[k][i], (float)cy);
       ea.z = atan2(M[j][i], M[i][i]);
   } else {
       ea.x = atan2(-M[j][k], M[j][j]);
       ea.y = atan2(-M[k][i], (float)cy);
       ea.z = 0;
   }
    }
    if (n==EulParOdd) {ea.x = -ea.x; ea.y = - ea.y; ea.z = -ea.z;}
    if (f==EulFrmR) {float t = ea.x; ea.x = ea.z; ea.z = t;}
    ea.w = order;
    return (ea);
}

/* Convert quaternion to Euler angles (in radians). */
EulerAngles Eul_FromQuat(Quat q, int order)
{
    HMatrix M;
    double Nq = q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    double s = (Nq > 0.0) ? (2.0 / Nq) : 0.0;
    double xs = q.x*s,	  ys = q.y*s,	 zs = q.z*s;
    double wx = q.w*xs,	  wy = q.w*ys,	 wz = q.w*zs;
    double xx = q.x*xs,	  xy = q.x*ys,	 xz = q.x*zs;
    double yy = q.y*ys,	  yz = q.y*zs,	 zz = q.z*zs;
    M[X][X]=(float)(1.0-(yy+zz)); M[X][Y]=(float)(xy - wz);     M[X][Z]=(float)(xz + wy);
    M[Y][X]=(float)(xy + wz);     M[Y][Y]=(float)(1.0-(xx+zz)); M[Y][Z]=(float)(yz - wx);
    M[Z][X]=(float)(xz - wy);     M[Z][Y]=(float)(yz + wx);     M[Z][Z]=(float)(1.0-(xx+yy));
    M[W][X]=M[W][Y]=M[W][Z]=M[X][W]=M[Y][W]=M[Z][W]=0.0; M[W][W]=1.0;
    return (Eul_FromHMatrix(M, order));
}
