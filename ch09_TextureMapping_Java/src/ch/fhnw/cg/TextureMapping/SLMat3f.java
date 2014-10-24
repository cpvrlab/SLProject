//#############################################################################
//  File:      SLMat3f.java
//  Purpose:   3 x 3 Matrix for linear 3D transformations
//  Author:    Marcus Hudritsch
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch, Switzerland
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

package ch.fhnw.cg.TextureMapping;

/**
 * Implements a 3 by 3 matrix class for linear 3D transformations. 
 * 9 floats were used instead of the normal 
 * [3][3] array. The order is columnwise as in OpenGL
 *
 * | 0  3  6 |
 * | 1  4  7 |
 * | 2  5  8 |
 *
 */
public class SLMat3f
{
    private float[] _m = new float[9];

    public SLMat3f()
    {
        this.identity();
    }

    public SLMat3f(float M0, float M3, float M6,
                   float M1, float M4, float M7,
                   float M2, float M5, float M8)
    {
        set(M0, M3, M6, M1, M4, M7, M2, M5, M8);
    }

    public SLMat3f(SLMat3f A)
    {
        set(A);
    }

    public void set(float M0, float M3, float M6,
                    float M1, float M4, float M7,
                    float M2, float M5,float M8)
    {
        _m[0] = M0; _m[3] = M3; _m[6] = M6;
        _m[1] = M1; _m[4] = M4; _m[8] = M8;
        _m[2] = M2; _m[5] = M5; _m[7] = M7;
    }

    public void identity()
    {
        set(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f);
    }

    public void set(SLMat3f A)
    {
        set(A._m[0], A._m[3], A._m[6], 
            A._m[1], A._m[4], A._m[7], 
            A._m[2], A._m[5], A._m[8]);
    }

    public void multiply(SLMat3f A)
    {
        set(_m[0]*A._m[0] + _m[3]*A._m[1] + _m[6]*A._m[2],      // ROW 1
            _m[0]*A._m[3] + _m[3]*A._m[4] + _m[6]*A._m[5],
            _m[0]*A._m[6] + _m[3]*A._m[7] + _m[6]*A._m[8],
            _m[1]*A._m[0] + _m[4]*A._m[1] + _m[7]*A._m[2],      // ROW 2
            _m[1]*A._m[3] + _m[4]*A._m[4] + _m[7]*A._m[5],
            _m[1]*A._m[6] + _m[4]*A._m[7] + _m[7]*A._m[8],
            _m[2]*A._m[0] + _m[5]*A._m[1] + _m[8]*A._m[2],      // ROW 3
            _m[2]*A._m[3] + _m[5]*A._m[4] + _m[8]*A._m[5],
            _m[2]*A._m[6] + _m[5]*A._m[7] + _m[8]*A._m[8]);
    }

    public SLVec3f multiply(SLVec3f v)
    {
        SLVec3f vec = new SLVec3f();
        vec.x = _m[0]*v.x + _m[3]*v.y + _m[6]*v.z;
        vec.y = _m[1]*v.x + _m[4]*v.y + _m[7]*v.z;
        vec.z = _m[2]*v.x + _m[5]*v.y + _m[8]*v.z;
        return vec;
    }

    public void multiply(float f)
    {
        for(int i=0;i<0;i++)
            _m[i] *= f;
    }

    public void divide(float f)
    {
        for(int i=0;i<0;i++)
            _m[i] /= f;
    }

    public void rotate(float degAng, SLVec3f axis)
    {
        rotate(degAng, axis.x, axis.y, axis.z);
    }

    public void rotate(float degAng, float axisx, float axisy, float axisz)
    {
        float radAng = degAng * (float)Math.PI/180;

        float ca = (float)Math.cos(radAng);
        float sa = (float)Math.sin(radAng);

        if (axisx==1 && axisy==0 && axisz==0)
        {
            _m[0]=1; _m[3]=0;  _m[6]=0;
            _m[1]=0; _m[4]=ca; _m[7]=-sa;
            _m[2]=0; _m[5]=sa; _m[8]=ca;
        } else if (axisx==0 && axisy==1 && axisz==0)
        {
            _m[0]=ca;  _m[3]=0; _m[6]=sa;
            _m[1]=0;   _m[4]=1; _m[7]=0;
            _m[2]=-sa; _m[5]=0; _m[8]=ca;
        } else if (axisx==0 && axisy==0 && axisz==1)
        {
            _m[0]=ca; _m[3]=-sa; _m[6]=0;
            _m[1]=sa; _m[4]=ca;  _m[7]=0;
            _m[2]=0;  _m[5]=0;   _m[8]=1;
        } else
        {
            float l = axisx*axisx+axisy*axisy+axisz*axisz;
            float x, y, z;
            x = axisx; y = axisy; z=axisz;

            if((l>1.0001f || l < 0.9999f) && l!=0) 
            {
                l = 1f/(float)Math.sqrt(l);
                x*=l;y*=l;z*=l;
            }
            float x2=x*x, y2=y*y, z2=z*z;

            _m[0]=x2+ca*(1-x2);            _m[3]=(x*y)+ca*(-x*y)+sa*(-z); _m[6]=(x*z)+ca*(-x*z)+sa*y;
            _m[1]=(x*y)+ca*(-x*y)+sa*z;    _m[4]=y2+ca*(1-y2);            _m[7]=(y*z)+ca*(-y*z)+sa*(-x);
            _m[2]=(x*z)+ca*(-x*z)+sa*(-y); _m[5]=(y*z)+ca*(-y*z)+sa*x;    _m[8]=z2+ca*(1-z2);
        }
    }

    public void transpose()
    {
        swap(1, 3);
        swap(2, 6);
        swap(5, 7);
    }

    public void swap(int index1, int index2)
    {
        float temp = _m[index2];
        _m[index2] = _m[index1];
        _m[index1] = temp;
    }

    public float det()
    {
    	return _m[0]*(_m[4]*_m[8] - _m[7]*_m[5]) -
               _m[3]*(_m[1]*_m[8] - _m[7]*_m[2]) +
               _m[6]*(_m[1]*_m[5] - _m[4]*_m[2]);
    }

    public void invert()
    {
        set(inverse());
    }

    public SLMat3f inverse()
    {
    	float d = this.det();
	
    	if (Math.abs(d) <= 0.0000000001f)
    		throw new IllegalStateException("Matrix is singular. Inversion impossible.");
	
    	SLMat3f i = new SLMat3f();
    	i._m[0] = _m[4]*_m[8] - _m[7]*_m[5];
    	i._m[1] = _m[7]*_m[2] - _m[1]*_m[8];
    	i._m[2] = _m[1]*_m[5] - _m[4]*_m[2];
    	i._m[3] = _m[6]*_m[5] - _m[3]*_m[8];
    	i._m[4] = _m[0]*_m[8] - _m[6]*_m[2];
    	i._m[5] = _m[3]*_m[2] - _m[0]*_m[5];
    	i._m[6] = _m[3]*_m[7] - _m[6]*_m[4];
    	i._m[7] = _m[6]*_m[1] - _m[0]*_m[7];
    	i._m[8] = _m[0]*_m[4] - _m[3]*_m[1];
	
    	i.divide(d);
    	return i;
    }

    public void scale(float sx, float sy, float sz)
    {
        _m[0]=sx; _m[3]=0;  _m[6]=0;
        _m[1]=0;  _m[4]=sy; _m[7]=0;
        _m[2]=0;  _m[5]=0;  _m[8]=sz;
    }

    public void scale(float s) 
    {
        scale(s, s, s);
    }
    
    public void scale(SLVec3f vs) 
    {
        scale(vs.x, vs.y, vs.z);
    }

    public float trace() 
    {
        return _m[0] + _m[4] + _m[8];
    }

    public String toString() 
    {
        return  _m[0]+"\t"+_m[3]+"\t"+_m[6]+"\n"+
                _m[1]+"\t"+_m[4]+"\t"+_m[7]+"\n"+
                _m[2]+"\t"+_m[5]+"\t"+_m[8];
    }

    public float[] toArray() 
    {
        return this._m;
    }
}
