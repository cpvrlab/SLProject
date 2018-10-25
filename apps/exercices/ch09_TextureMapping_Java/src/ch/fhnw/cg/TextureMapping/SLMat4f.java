//#############################################################################
//  File:      Globals/SLMat4f.java
//  Purpose:   4 x 4 Matrix for affine transformations
//  Author:    Marcus Hudritsch
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch, Switzerland
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

package ch.fhnw.cg.TextureMapping;


/**
 * Implements a 4 by 4 matrix class for affine transformations.
 * 16 floats were used instead of the normal[4][4] to be compliant with OpenGL. 
 * OpenGL uses premultiplication with column vectors. These matrices can be fed 
 * directly into the OpenGL matrix stack with glLoadMatrix or glMultMatrix. The
 * index layout is as follows:
 * 
 *      | 0  4  8 12 |
 *      | 1  5  9 13 |
 *  M = | 2  6 10 14 |
 *      | 3  7 11 15 |
 *      
 * Vectors are interpreted as column vectors when applying matrix multiplications. 
 * This means a vector is as a single column, 4-row matrix. The result is that the 
 * transformations implemented by the matrices happens right-to-left e.g. if 
 * vector V is to be transformed by M1 then M2 then M3, the calculation would 
 * be M3 * M2 * M1 * V. The order that matrices are concatenated is vital 
 * since matrix multiplication is not commutative, i.e. you can get a different 
 * result if you concatenate in the wrong order.
 * The use of column vectors and right-to-left ordering is the standard in most 
 * mathematical texts, and is the same as used in OpenGL. It is, however, the 
 * opposite of Direct3D, which has inexplicably chosen to differ from the 
 * accepted standard and uses row vectors and left-to-right matrix multiplication.
 */
public class SLMat4f 
{
    private float[] _m = new float[16];
    
    
    public SLMat4f(float M0, float M4, float M8,  float M12, 
    		       float M1, float M5, float M9,  float M13, 
    		       float M2, float M6, float M10, float M14,
                   float M3, float M7, float M11, float M15) 
    {
        _m[0] = M0; _m[4] = M4; _m[8]  = M8;  _m[12] = M12;
        _m[1] = M1; _m[5] = M5; _m[9]  = M9;  _m[13] = M13;
        _m[2] = M2; _m[6] = M6; _m[10] = M10; _m[14] = M14;
        _m[3] = M3; _m[7] = M7; _m[11] = M11; _m[15] = M15;
    }
    
    public SLMat4f() {}
    
    public SLMat4f(SLMat4f mat) 
    {
        set(mat);
    }
    
    public void identity() 
    {
        _m[0] = 1f; _m[4] = 0f; _m[ 8] = 0f; _m[12] = 0f;
        _m[1] = 0f; _m[5] = 1f; _m[ 9] = 0f; _m[13] = 0f;
        _m[2] = 0f; _m[6] = 0f; _m[10] = 1f; _m[14] = 0f;
        _m[3] = 0f; _m[7] = 0f; _m[11] = 0f; _m[15] = 1f;
    }

    public void set(SLMat4f A) 
    {  
        _m[0]=A._m[0]; _m[4]=A._m[4]; _m[ 8]=A._m[8];  _m[12]=A._m[12];
        _m[1]=A._m[1]; _m[5]=A._m[5]; _m[ 9]=A._m[9];  _m[13]=A._m[13];
        _m[2]=A._m[2]; _m[6]=A._m[6]; _m[10]=A._m[10]; _m[14]=A._m[14];
        _m[3]=A._m[3]; _m[7]=A._m[7]; _m[11]=A._m[11]; _m[15]=A._m[15];
    }

    public void set(float M0, float M4, float M8,  float M12, 
    		        float M1, float M5, float M9,  float M13, 
    		        float M2, float M6, float M10, float M14,
                    float M3, float M7, float M11, float M15) 
    {
        _m[0] = M0; _m[4] = M4; _m[8] = M8;   _m[12] = M12;
        _m[1] = M1; _m[5] = M5; _m[9] = M9;   _m[13] = M13;
        _m[2] = M2; _m[6] = M6; _m[10] = M10; _m[14] = M14;
        _m[3] = M3; _m[7] = M7; _m[11] = M11; _m[15] = M15;
    }

    public void set(int index, float f) 
    {
        if(index < 0 || index > 15 )
            throw new IndexOutOfBoundsException("Mat4 has 16 fields");
        _m[index] = f;
    }
    
    public float get(int index) 
    {
        if(index < 0 || index > 15 )
            throw new IndexOutOfBoundsException("Mat4 has 16 fields");
        return _m[index];
    }
    
    public void multiply(SLMat4f A) 
    {
        set(_m[0]*A._m[ 0] + _m[4]*A._m[ 1] + _m[8] *A._m[ 2] + _m[12]*A._m[ 3], //row 1
            _m[0]*A._m[ 4] + _m[4]*A._m[ 5] + _m[8] *A._m[ 6] + _m[12]*A._m[ 7],
            _m[0]*A._m[ 8] + _m[4]*A._m[ 9] + _m[8] *A._m[10] + _m[12]*A._m[11],
            _m[0]*A._m[12] + _m[4]*A._m[13] + _m[8] *A._m[14] + _m[12]*A._m[15],
              
            _m[1]*A._m[ 0] + _m[5]*A._m[ 1] + _m[9] *A._m[ 2] + _m[13]*A._m[ 3], //row 2
            _m[1]*A._m[ 4] + _m[5]*A._m[ 5] + _m[9] *A._m[ 6] + _m[13]*A._m[ 7],
            _m[1]*A._m[ 8] + _m[5]*A._m[ 9] + _m[9] *A._m[10] + _m[13]*A._m[11],
            _m[1]*A._m[12] + _m[5]*A._m[13] + _m[9] *A._m[14] + _m[13]*A._m[15],
              
            _m[2]*A._m[ 0] + _m[6]*A._m[ 1] + _m[10]*A._m[ 2] + _m[14]*A._m[ 3], //row 3
            _m[2]*A._m[ 4] + _m[6]*A._m[ 5] + _m[10]*A._m[ 6] + _m[14]*A._m[ 7],
            _m[2]*A._m[ 8] + _m[6]*A._m[ 9] + _m[10]*A._m[10] + _m[14]*A._m[11],
            _m[2]*A._m[12] + _m[6]*A._m[13] + _m[10]*A._m[14] + _m[14]*A._m[15],
              
            _m[3]*A._m[ 0] + _m[7]*A._m[ 1] + _m[11]*A._m[ 2] + _m[15]*A._m[ 3], //row 4
            _m[3]*A._m[ 4] + _m[7]*A._m[ 5] + _m[11]*A._m[ 6] + _m[15]*A._m[ 7],
            _m[3]*A._m[ 8] + _m[7]*A._m[ 9] + _m[11]*A._m[10] + _m[15]*A._m[11],
            _m[3]*A._m[12] + _m[7]*A._m[13] + _m[11]*A._m[14] + _m[15]*A._m[15]);
    }
    
    public SLVec3f multiply(SLVec3f v) 
    {
        float W = _m[3]*v.x + _m[7]*v.y + _m[11]*v.z + _m[15];
        return new SLVec3f((_m[0]*v.x + _m[4]*v.y + _m[ 8]*v.z + _m[12]) / W,
                           (_m[1]*v.x + _m[5]*v.y + _m[ 9]*v.z + _m[13]) / W,
                           (_m[2]*v.x + _m[6]*v.y + _m[10]*v.z + _m[14]) / W);
    }
    
    public SLVec4f multiply(SLVec4f v) 
    {
        SLVec4f newV = new SLVec4f(_m[0]*v.x + _m[4]*v.y + _m[ 8]*v.z + _m[12]*v.w,
                                   _m[1]*v.x + _m[5]*v.y + _m[ 9]*v.z + _m[13]*v.w,
                                   _m[2]*v.x + _m[6]*v.y + _m[10]*v.z + _m[14]*v.w,
                                   _m[3]*v.x + _m[7]*v.y + _m[11]*v.z + _m[15]*v.w); 
        return newV;
    }    
    
    public SLVec3f translation() 
    {
        return new SLVec3f(_m[12], _m[13], _m[14]);
    }
    
    public SLMat4f inverse() 
    {
        SLMat4f I = new SLMat4f();

        // Code from Mesa-2.2\src\glu\project.c
        float det, d12, d13, d23, d24, d34, d41;

        // Inverse = adjoint / det. (See linear algebra texts.)
        // pre-compute 2x2 dets for last two rows when computing
        // cof_actors of first two rows.
        d12 = (_m[ 2] * _m[ 7] - _m[ 3] * _m[ 6]);
        d13 = (_m[ 2] * _m[11] - _m[ 3] * _m[10]);
        d23 = (_m[ 6] * _m[11] - _m[ 7] * _m[10]);
        d24 = (_m[ 6] * _m[15] - _m[ 7] * _m[14]);
        d34 = (_m[10] * _m[15] - _m[11] * _m[14]);
        d41 = (_m[14] * _m[ 3] - _m[15] * _m[ 2]);

        I._m[0] =  (_m[5] * d34 - _m[9] * d24 + _m[13] * d23);
        I._m[1] = -(_m[1] * d34 + _m[9] * d41 + _m[13] * d13);
        I._m[2] =  (_m[1] * d24 + _m[5] * d41 + _m[13] * d12);
        I._m[3] = -(_m[1] * d23 - _m[5] * d13 + _m[ 9] * d12);

        // Compute determinant as early as possible using these cof_actors.
        det = _m[0] * I._m[0] + _m[4] * I._m[1] + _m[8] * I._m[2] + _m[12] * I._m[3];

        // Run singularity test.
        if (Math.abs(det) <= 0.00005) 
            throw new IllegalArgumentException("Matrix is singular. Inversion impossible.");
        else 
        {
            float invDet = 1 / det;
            // Compute rest of inverse.
            I._m[0] *= invDet;
            I._m[1] *= invDet;
            I._m[2] *= invDet;
            I._m[3] *= invDet;

            I._m[4] = -(_m[4] * d34 - _m[8] * d24 + _m[12] * d23) * invDet;
            I._m[5] = (_m[0] * d34 + _m[8] * d41 + _m[12] * d13) * invDet;
            I._m[6] = -(_m[0] * d24 + _m[4] * d41 + _m[12] * d12) * invDet;
            I._m[7] = (_m[0] * d23 - _m[4] * d13 + _m[8] * d12) * invDet;

            // Pre-compute 2x2 dets for first two rows when computing
            // cofactors of last two rows.
            d12 = _m[0] * _m[5] - _m[1] * _m[4];
            d13 = _m[0] * _m[9] - _m[1] * _m[8];
            d23 = _m[4] * _m[9] - _m[5] * _m[8];
            d24 = _m[4] * _m[13] - _m[5] * _m[12];
            d34 = _m[8] * _m[13] - _m[9] * _m[12];
            d41 = _m[12] * _m[1] - _m[13] * _m[0];

            I._m[8] = (_m[7] * d34 - _m[11] * d24 + _m[15] * d23) * invDet;
            I._m[9] = -(_m[3] * d34 + _m[11] * d41 + _m[15] * d13) * invDet;
            I._m[10] = (_m[3] * d24 + _m[7] * d41 + _m[15] * d12) * invDet;
            I._m[11] = -(_m[3] * d23 - _m[7] * d13 + _m[11] * d12) * invDet;
            I._m[12] = -(_m[6] * d34 - _m[10] * d24 + _m[14] * d23) * invDet;
            I._m[13] = (_m[2] * d34 + _m[10] * d41 + _m[14] * d13) * invDet;
            I._m[14] = -(_m[2] * d24 + _m[6] * d41 + _m[14] * d12) * invDet;
            I._m[15] = (_m[2] * d23 - _m[6] * d13 + _m[10] * d12) * invDet;
        }
        return I;
    }
    
    public SLMat3f inverseTransposed()
    {
    	SLMat3f mat3 = new SLMat3f(_m[0], _m[4], _m[ 8], 
					               _m[1], _m[5], _m[ 9], 
					               _m[2], _m[6], _m[10]);
    	mat3.invert();
    	mat3.transpose();
    	return mat3;
    }
    
    public void translate(SLVec3f t) 
    {
        multiply(SLMat4f.translation(t));
    }
    
    public void translate(float x, float y, float z) 
    {
        multiply(SLMat4f.translation(new SLVec3f(x, y, z)));
    }
    
    public void rotate(float degAng, SLVec3f axis) 
    {
        rotate(degAng, axis.x, axis.y, axis.z);
    }
    
    public void rotate(float degAng, float axisx, float axisy, float axisz) 
    {
        multiply(SLMat4f.rotation(degAng, axisx, axisy, axisz));
    }

    public void scale(SLVec3f s) 
    {
        scale(s.x, s.y, s.z);
    }
    
    public void scale(float sx, float sy, float sz) 
    {
        multiply(SLMat4f.scaling(sx, sy, sz));
    }
    
    public void invert() 
    {
        set(inverse());
    }
    
    public void transpose() 
    {
        swap(1, 4);
        swap(2, 8);
        swap(6, 9);
        swap(3, 12);
        swap(7, 13);
        swap(11, 14);
    }
    
    public void swap(int i1, int i2) 
    {
        float temp = get(i1);
        set(i1, get(i2));
        set(i2, temp);
    }

    public void posAtUp(SLVec3f pos) 
    {
        posAtUp(pos, new SLVec3f(), new SLVec3f());
    }

    public void posAtUp(SLVec3f pos, SLVec3f dirAt, SLVec3f dirUp) 
    {
        lightAt(pos, dirAt, dirUp);
    }
    
    public void lightAt(SLVec3f pos, SLVec3f dirAt, SLVec3f dirUp) 
    {
        SLVec3f VX = new SLVec3f();
        SLVec3f VY = new SLVec3f();
        SLVec3f VZ;
        //SLVec3f VT = new SLVec3f();
        
        SLMat3f xz = new SLMat3f(0f, 0f, 1f,
                                 0f, 0f, 0f,
                                -1f, 0f, 0f);
        
        VZ = SLVec3f.subtract(pos, dirAt);
        if(dirUp.isApproxEqual(SLVec3f.ZERO, 0f)) 
        {
            VX = xz.multiply(VZ);
            VX.normalize();
            VY = SLVec3f.cross(VZ, VX);
            VY.normalize();
        } else 
        {
            VX = SLVec3f.cross(dirUp, VZ);
            VX.normalize();
            VY = SLVec3f.cross(VZ, VX);
            VY.normalize();
        }
        
        set(VX.x, VY.x, VZ.x, pos.x, 
            VX.y, VY.y, VZ.y, pos.y,
            VX.z, VY.z, VZ.z, pos.z,
            0f, 0f, 0f, 1f);
    }
    
    public static SLMat4f translation(SLVec3f t) 
    {
        SLMat4f tr = new SLMat4f();
        tr.identity();
        tr.set(12, t.x);
        tr.set(13, t.y);
        tr.set(14, t.z);
        return tr;
    }
    
    public static SLMat4f scaling(float sx, float sy, float sz) 
    {
        SLMat4f s = new SLMat4f();
        s.identity();
        s.set(0, sx);
        s.set(5, sy);
        s.set(10, sy);
        return s;
    }
    
    public static SLMat4f rotation(float degAng, float axisx, float axisy, float axisz) 
    {  
        SLMat4f r = new SLMat4f();
        float RadAng = (float) Math.toRadians(degAng);
        float ca = (float) Math.cos(RadAng);
        float sa = (float) Math.sin(RadAng);

        if (axisx == 1 && axisy == 0 && axisz == 0) // about x-axis
        {
            r._m[0] = 1;
            r._m[4] = 0;
            r._m[8] = 0;
            r._m[1] = 0;
            r._m[5] = ca;
            r._m[9] = -sa;
            r._m[2] = 0;
            r._m[6] = sa;
            r._m[10] = ca;
        } else if (axisx == 0 && axisy == 1 && axisz == 0) // about y-axis
        {
            r._m[0] = ca;
            r._m[4] = 0;
            r._m[8] = sa;
            r._m[1] = 0;
            r._m[5] = 1;
            r._m[9] = 0;
            r._m[2] = -sa;
            r._m[6] = 0;
            r._m[10] = ca;
        } else if (axisx == 0 && axisy == 0 && axisz == 1) // about z-axis
        {
            r._m[0] = ca;
            r._m[4] = -sa;
            r._m[8] = 0;
            r._m[1] = sa;
            r._m[5] = ca;
            r._m[9] = 0;
            r._m[2] = 0;
            r._m[6] = 0;
            r._m[10] = 1;
        } else // arbitrary axis
        {
            float len = axisx * axisx + axisy * axisy + axisz * axisz; // length
                                                                        // squared
            float x, y, z;
            x = axisx;
            y = axisy;
            z = axisz;
            if (len > 1.0001 || len < 0.9999 && len != 0) 
            {
                len = 1 / (float) Math.sqrt(len);
                x *= len;
                y *= len;
                z *= len;
            }
            float xy = x * y, yz = y * z, xz = x * z, xx = x * x, yy = y * y, zz = z
                    * z;
            r._m[0] = xx + ca * (1 - xx);
            r._m[4] = xy - xy * ca - z * sa;
            r._m[8] = xz - xz * ca + y * sa;
            r._m[1] = xy - xy * ca + z * sa;
            r._m[5] = yy + ca * (1 - yy);
            r._m[9] = yz - yz * ca - x * sa;
            r._m[2] = xz - xz * ca - y * sa;
            r._m[6] = yz - yz * ca + x * sa;
            r._m[10] = zz + ca * (1 - zz);
        }
        r._m[3] = r._m[7] = r._m[11] = 0;
        r._m[15] = 1;

        return r;
    }
    
    public void frustum(float l, float r, float b, float t, float n, float f) 
    {
    	_m[0]=(2*n)/(r-l); _m[4]=0;           _m[8] = (r+l)/(r-l); _m[12]=0;
    	_m[1]=0;           _m[5]=(2*n)/(t-b); _m[9] = (t+b)/(t-b); _m[13]=0;
    	_m[2]=0;           _m[6]=0;           _m[10]=-(f+n)/(f-n); _m[14]=(-2*f*n)/(f-n);
    	_m[3]=0;           _m[7]=0;           _m[11]=-1;           _m[15]=0;
    }
    
    public void perspective(float fov, float aspect, float n, float f) 
    {  
        float t = (float)(Math.tan(Math.toRadians(fov *0.5))*n);
        float b = -t;
        float r = t*aspect;
        float l = -r;
        frustum(l,r,b,t,n,f);
    }
    
    public void ortho(float l, float r, float b, float t, float n, float f)
	{  	
    	_m[0]=2/(r-l); _m[4]=0;       _m[8]=0;         _m[12]=-(r+l)/(r-l);
		_m[1]=0;       _m[5]=2/(t-b); _m[9]=0;         _m[13]=-(t+b)/(t-b);
		_m[2]=0;       _m[6]=0;       _m[10]=-2/(f-n); _m[14]=-(f+n)/(f-n);
		_m[3]=0;       _m[7]=0;       _m[11]=0;        _m[15]=1;
	}

    public void viewport(float x,  float y, float ww, float wh, float n,  float f) 
    {
        float ww2 = ww*0.5f;
        float wh2 = wh*0.5f;

        // negate the first wh2 because windows has topdown window coords
        _m[0]=ww2; _m[4]=0;    _m[8] =0;          _m[12]=x+ww2;
        _m[1]=0;   _m[5]=-wh2; _m[9] =0;          _m[13]=y+wh2;
        _m[2]=0;   _m[6]=0;    _m[10]=(f-n)*0.5f; _m[14]=(f+n)*0.5f;
        _m[3]=0;   _m[7]=0;    _m[11]=0;          _m[15]=1;
    }
    
    public SLMat3f mat3()
    {
        SLMat3f mat3 = new SLMat3f(_m[0], _m[4], _m[ 8], 
                                   _m[1], _m[5], _m[ 9], 
                                   _m[2], _m[6], _m[10]);
        return mat3;
    }
    
    @Override
    public String toString()
    {
        StringBuffer sb = new StringBuffer();
        String format = "%01.02f";
        sb.append(String.format(format, _m[0])+", ");
        sb.append(String.format(format, _m[4])+", ");
        sb.append(String.format(format, _m[8])+", ");
        sb.append(String.format(format, _m[12])+", \n");
        sb.append(String.format(format, _m[1])+", ");
        sb.append(String.format(format, _m[5])+", ");
        sb.append(String.format(format, _m[9])+", ");
        sb.append(String.format(format, _m[13])+", \n");
        sb.append(String.format(format, _m[2])+", ");
        sb.append(String.format(format, _m[6])+", ");
        sb.append(String.format(format, _m[10])+", ");
        sb.append(String.format(format, _m[14])+", \n");
        sb.append(String.format(format, _m[3])+", ");
        sb.append(String.format(format, _m[7])+", ");
        sb.append(String.format(format, _m[11])+", ");
        sb.append(String.format(format, _m[15])+", \n");
        return sb.toString();
    }

    public float[] toArray() 
    {
        return this._m;
    }
}
