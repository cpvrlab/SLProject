//#############################################################################
//  File:      SLVec4f.java
//  Purpose:   4 Component vector class
//  Date:      September 2012 (HS12)
//  Copyright: Marcus Hudritsch, Switzerland
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

package ch.fhnw.cg.TextureMapping;

public class SLVec4f 
{
	/**
	 * Predefined Vectors
	 */
	
	/**
	 * Coordinates
	 */
	public float x;
	public float y;
	public float z;
	public float w;
	
	//Constructors
	
	public SLVec4f(float x, float y, float z, float w) 
    {
		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;
	}
	
	public SLVec4f(SLVec4f vector) 
    {
		this.x = vector.x;
		this.y = vector.y;
		this.z = vector.z;
		this.w = vector.w;
	}
	
	public SLVec4f() 
    {
		this(0f, 0f, 0f, 0f);
	}
	
	public SLVec4f(float[] coordinates) 
    {
		if(coordinates==null || coordinates.length<4) 
        {
			throw new IllegalArgumentException("Coordinate Array must contain at least 4 dimensions");
		}
		this.x = coordinates[0];
		this.y = coordinates[1];
		this.z = coordinates[2];
		this.w = coordinates[3];
	}
	
	public SLVec4f add(SLVec4f vec) 
    {
		this.x += vec.x;
		this.y += vec.y;
		this.z += vec.z;
		this.w += vec.w;
		return this;
	}

	public SLVec4f add(float s) 
    {
		this.x += s;
		this.y += s;
		this.z += s;
		this.w += w;
		return this;
	}
	
	public static SLVec4f add(SLVec4f vec1, SLVec4f vec2) 
    {
		return new SLVec4f(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z, vec1.w+vec2.w);
	}
	
	public static SLVec4f add(SLVec4f vec1, float s) 
    {
		return new SLVec4f(vec1.x + s, vec1.y + s, vec1.z + s, vec1.w + s);
	}
	
	
	public SLVec4f subtract(SLVec4f vec) 
    {
		this.x -= vec.x;
		this.y -=vec.y;
		this.z -=vec.z;
		this.w -=vec.w;
		return this;
	}
	
	public SLVec4f subtract(float s) 
    {
		this.x -= s;
		this.y -= s;
		this.z -= s;
		this.w -= s;
		return this;
	}
	
	public static SLVec4f subtract(SLVec4f vec1, SLVec4f vec2) 
    {
		return new SLVec4f(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z, vec1.w - vec2.w);
	}
	
	public static SLVec4f subtract(SLVec4f vec1, float s) 
    {
		return new SLVec4f(vec1.x - s, vec1.y - s, vec1.z - s, vec1.w - s);
	}
	

	public SLVec4f divide(SLVec4f vec) 
    {
		if(vec.x==0f || vec.y==0f || vec.z==0f || vec.w==0f)
			throw new IllegalArgumentException("Division by 0 not allowed!");
		this.x /= vec.x;
		this.y /=vec.y;
		this.z /=vec.z;
		this.w /=vec.w;
		return this;
	}
	
	public SLVec4f divide(float s) 
    {
		if(s==0f)
			throw new IllegalArgumentException("Division by 0 not allowed!");
		this.x /= s;
		this.y /= s;
		this.z /= s;
		this.w /= s;
		return this;
	}
	
	public static SLVec4f divide(SLVec4f vec1, SLVec4f vec2) 
    {
		if(vec2.x==0f || vec2.y==0f || vec2.z==0f)
			throw new IllegalArgumentException("Division by 0 not allowed!");
		return new SLVec4f(vec1.x / vec2.x, vec1.y / vec2.y, vec1.z / vec2.z, vec1.w/vec2.w);
	}
	
	public static SLVec4f divide(SLVec4f vec1, float s) 
    {
		if(s==0f)
			throw new IllegalArgumentException("Division by 0 not allowed!");
		return new SLVec4f(vec1.x / s, vec1.y / s, vec1.z / s, vec1.w / s);
	}
	
	
	public SLVec4f multiply(SLVec4f vec) 
    {
		this.x *= vec.x;
		this.y *=vec.y;
		this.z *=vec.z;
		this.w *=vec.w;
		return this;
	}
	
	public SLVec4f multiply(float s) 
    {
		this.x *= s;
		this.y *= s;
		this.z *= s;
		this.w *= s;
		return this;
	}
	
	public static SLVec4f multiply(SLVec4f vec1, SLVec4f vec2) 
    {
		return new SLVec4f(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z, vec1.w * vec2.w);
	}
	
	public static SLVec4f multiply(SLVec4f vec1, float s) 
    {
		return new SLVec4f(vec1.x * s, vec1.y * s, vec1.z * s, vec1.w * s);
	}
	
	public float dot(SLVec4f vec) 
    {
		return SLVec4f.dot(this, vec);
	}
	
	public static float dot(SLVec4f u, SLVec4f v) 
    {
		return (u.x * v.x) + (u.y * v.y) + (u.z * v.z) + (u.w * v.w);
	}
	
	public SLVec4f cross(SLVec4f vec) 
    {
		return SLVec4f.cross(this, vec);
	}
	
	public static SLVec4f cross(SLVec4f u, SLVec4f v) 
    {
	    return new SLVec4f(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x, 1f);
	}
	
	public SLVec4f negate() 
    {
		return this.multiply(-1f);
	}
	
	public static SLVec4f negate(SLVec4f vec) 
    {
		return vec.multiply(-1f);
	}
	
	public boolean isApproxEqual(SLVec4f vec, float tolerance) 
    {
		return Math.abs(this.x-vec.x) <= tolerance && Math.abs(this.y-vec.y) <= tolerance && Math.abs(this.z-vec.z) <= tolerance && Math.abs(this.w-vec.w) <= tolerance;
	}
	
	public float length() 
    {
	      return (float)Math.sqrt(this.x*this.x + this.y*this.y + this.z*this.z + this.w*this.w);
	}
	public float length2() 
    {
	      return (this.x*this.x + this.y*this.y + this.z*this.z + this.w*this.w);
	}
	
	public SLVec4f normalize() 
    {
		return this.divide(length());
	}

	public float[] toArray() 
    {
		return new float[] {x, y, z, w};
	}
	
	@Override
	public boolean equals(Object o) 
    {
		if(o instanceof SLVec4f)
			return isApproxEqual((SLVec4f)o, 0.0f);
		return false;
	}
	
	@Override
	public Object clone() 
    {
		return new SLVec4f(this);
	}
	
	public static float[] toArray(SLVec4f[] vecs) 
    {
		float[] arr = new float[vecs.length*4];
		for(int i=0;i<vecs.length;i++) 
        {
			SLVec4f v = vecs[i];
			arr[i*4+0] = v.x;
			arr[i*4+1] = v.y;
			arr[i*4+2] = v.z;
			arr[i*4+3] = v.w;
		}
		return arr;	
	}
}
