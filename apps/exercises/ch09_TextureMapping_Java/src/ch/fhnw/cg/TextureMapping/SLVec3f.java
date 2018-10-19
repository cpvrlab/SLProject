//#############################################################################
//  File:      SLVec3f.java
//  Purpose:   3 Component vector class
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch, Switzerland
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

package ch.fhnw.cg.TextureMapping;


public class SLVec3f 
{
    /**
     * Predefined Vectors
     */
    public static final SLVec3f ZERO  = new SLVec3f(0f, 0f, 0f);
    public static final SLVec3f XAxis = new SLVec3f(1f, 0f, 0f);
    public static final SLVec3f YAxis = new SLVec3f(0f, 1f, 0f);
    public static final SLVec3f ZAxis = new SLVec3f(0f, 0f, 1f);
    
    /**
     * Coordinates
     */
    public float x;
    public float y;
    public float z;
    
    //Constructors
    
    public SLVec3f(float x, float y, float z) 
    {
        this.x = x;
        this.y = y;
        this.z = z;
    }
    
    public SLVec3f(SLVec2f vector) 
    {
        this.x = vector.x;
        this.y = vector.y;
        this.z = 0;
    }

    public SLVec3f(SLVec3f vector) 
    {
        this.x = vector.x;
        this.y = vector.y;
        this.z = vector.z;
    }
    
    public SLVec3f() 
    {
        this(SLVec3f.ZERO);
    }
    
    public SLVec3f(float[] coordinates) 
    {
        if(coordinates==null || coordinates.length<3)
            throw new IllegalArgumentException("Coordinate Array must contain at least 3 dimensions");
        this.x = coordinates[0];
        this.y = coordinates[1];
        this.z = coordinates[2];
    }
    
    public SLVec3f add(SLVec3f vec) 
    {
        this.x += vec.x;
        this.y += vec.y;
        this.z += vec.z;
        return this;
    }

    public SLVec3f add(float s) 
    {
        this.x += s;
        this.y += s;
        this.z += s;
        return this;
    }
    
    public static SLVec3f add(SLVec3f vec1, SLVec3f vec2) 
    {
        return new SLVec3f(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z);
    }
    
    public static SLVec3f add(SLVec3f vec1, float s) 
    {
        return new SLVec3f(vec1.x + s, vec1.y + s, vec1.z + s);
    }
    
    
    public SLVec3f subtract(SLVec3f vec) 
    {
        this.x -= vec.x;
        this.y -=vec.y;
        this.z -=vec.z;
        return this;
    }
    
    public SLVec3f subtract(float s) 
    {
        this.x -= s;
        this.y -= s;
        this.z -= s;
        return this;
    }
    
    public static SLVec3f subtract(SLVec3f vec1, SLVec3f vec2) 
    {
        return new SLVec3f(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z);
    }
    
    public static SLVec3f subtract(SLVec3f vec1, float s) 
    {
        return new SLVec3f(vec1.x - s, vec1.y - s, vec1.z - s);
    }
    

    public SLVec3f divide(SLVec3f vec) 
    {
        if(vec.x==0f || vec.y==0f || vec.z==0f)
            throw new IllegalArgumentException("Division by 0 not allowed!");
        this.x /= vec.x;
        this.y /=vec.y;
        this.z /=vec.z;
        return this;
    }
    
    public SLVec3f divide(float s) 
    {
        if(s==0f)
            throw new IllegalArgumentException("Division by 0 not allowed!");
        this.x /= s;
        this.y /= s;
        this.z /= s;
        return this;
    }
    
    public static SLVec3f divide(SLVec3f vec1, SLVec3f vec2) 
    {
        if(vec2.x==0f || vec2.y==0f || vec2.z==0f)
            throw new IllegalArgumentException("Division by 0 not allowed!");
        return new SLVec3f(vec1.x / vec2.x, vec1.y / vec2.y, vec1.z / vec2.z);
    }
    
    public static SLVec3f divide(SLVec3f vec1, float s) 
    {
        if(s==0f)
            throw new IllegalArgumentException("Division by 0 not allowed!");
        return new SLVec3f(vec1.x / s, vec1.y / s, vec1.z / s);
    }
    
    
    public SLVec3f multiply(SLVec3f vec) 
    {
        this.x *= vec.x;
        this.y *=vec.y;
        this.z *=vec.z;
        return this;
    }
    
    public SLVec3f multiply(float s) 
    {
        this.x *= s;
        this.y *= s;
        this.z *= s;
        return this;
    }
    
    public static SLVec3f multiply(SLVec3f vec1, SLVec3f vec2) 
    {
        return new SLVec3f(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z);
    }
    
    public static SLVec3f multiply(SLVec3f vec1, float s) 
    {
        return new SLVec3f(vec1.x * s, vec1.y * s, vec1.z * s);
    }
    
    public float dot(SLVec3f vec) 
    {
        return SLVec3f.dot(this, vec);
    }
    
    public static float dot(SLVec3f u, SLVec3f v) 
    {
        return (u.x * v.x) + (u.y * v.y) + (u.z * v.z);
    }
    
    public SLVec3f cross(SLVec3f vec) 
    {
        return SLVec3f.cross(this, vec);
    }
    
    public static SLVec3f cross(SLVec3f u, SLVec3f v) 
    {
        return new SLVec3f(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x );
    }
    
    public SLVec3f negate() 
    {
        return this.multiply(-1f);
    }
    
    public static SLVec3f negate(SLVec3f vec) 
    {
        return vec.multiply(-1f);
    }
    
    public boolean isApproxEqual(SLVec3f vec, float tolerance) 
    {
        return Math.abs(this.x-vec.x) <= tolerance && Math.abs(this.y-vec.y) <= tolerance && Math.abs(this.z-vec.z) <= tolerance;
    }
    
    public float length() 
    {
        return (float)Math.sqrt(this.x*this.x + this.y*this.y + this.z*this.z);
    }
    public float length2() 
    {
        return (this.x*this.x + this.y*this.y + this.z*this.z);
    }
    
    public SLVec3f normalize() 
    {
        return this.divide(length());
    }

    public float[] toArray() 
    {
        return new float[] {x, y, z};
    }
    
    public SLVec2f toVec2() 
    {
        return new SLVec2f(x, y);
    }
    
    @Override
    public boolean equals(Object o) 
    {
        if(o instanceof SLVec3f)
            return isApproxEqual((SLVec3f)o, 0.0f);
        return false;
    }
    
    @Override
    public Object clone() 
    {
        return new SLVec3f(this);
    }
    
	public SLVec3f copy() 
	{
	    return new SLVec3f(this);
	}
    
    public static float[] toArray(SLVec3f[] vecs) 
    {
        float[] arr = new float[vecs.length*3];
        for(int i=0;i<vecs.length;i++) 
        {
            SLVec3f v = vecs[i];
            arr[i*3+0] = v.x;
            arr[i*3+1] = v.y;
            arr[i*3+2] = v.z;
        }
        return arr;    
    }
}
