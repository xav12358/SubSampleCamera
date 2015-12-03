#define BLK_SIZE    16
//prefix D for decoding
#define DSHRD_LEN   (BLK_SIZE/2)
#define DSHRD_SIZE  (2*DSHRD_LEN*DSHRD_LEN)

uchar4 convertYVUtoRGBA(int y, int u, int v)
{
    uchar4 ret;
    y-=16;
    u-=128;
    v-=128;
    int b = y + (int)(1.772f*u);
    int g = y - (int)(0.344f*u + 0.714f*v);
    int r = y + (int)(1.402f*v);
    int val = ((r+g+b)/3)>255?255:(r+g+b)/3;
    ret.x = val;
    ret.y = val;
    ret.z = val;
    ret.w = 255;
    return ret;
}

__kernel void nv21togray( __global uchar4* out,
                          __global uchar*  in,
                          int    im_width,
                          int    im_height)
{
    __local uchar uvShrd[DSHRD_SIZE];
    int gx	= get_global_id(0);
    int gy	= get_global_id(1);
    int lx  = get_local_id(0);
    int ly  = get_local_id(1);
    int off = im_width*im_height;
    // every thread whose
    // both x,y indices are divisible
    // by 2 move the u,v corresponding
    // to the 2x2 block into shared mem
    int inIdx= gy*im_width+gx;
    int uvIdx= off + (gy/2)*im_width + (gx & ~1);
    int shlx = lx/2;
    int shly = ly/2;
    int shIdx= 2*(shlx+shly*DSHRD_LEN);
    if( gx%2==0 && gy%2==0 ) {
        uvShrd[shIdx+0] = in[uvIdx+0];
        uvShrd[shIdx+1] = in[uvIdx+1];
    }
    // do some work while others copy
    // uv to shared memory
    int y   = (0xFF & ((int)in[inIdx]));
    if( y < 16 ) y=16;
    barrier(CLK_LOCAL_MEM_FENCE);
    // return invalid threads
    if( gx >= im_width || gy >= im_height )
        return;
    // convert color space
    int v   = (0xFF & ((int)uvShrd[shIdx+0]));
    int u   = (0xFF & ((int)uvShrd[shIdx+1]));
    // write output to image
    out[inIdx]  = convertYVUtoRGBA(y,u,v);
}


__kernel void downfilter_x_g( 
    __global uchar4 *src,
    __global uchar4 *dst, int w, int h )
{


    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    
    float x0 = src[ix-2+(iy)*w].x/16.0;
    float x1 = src[ix-1+(iy)*w].x/8.0;
    float x2 = 3*src[ix+(iy)*w].x/4.0;
    float x3 = src[ix+1+(iy)*w].x/8.0;
    float x4 = src[ix+2+(iy)*w].x/16.0;
    

    int output = round( x0 + x1 + x2 + x3 + x4 );
	if(output >255)
		output = 255;
    if( ix < w && iy < h ) {
        dst[iy*w + ix ] = (uchar4)(output,output,output,255);  // uncoalesced when writing to memory object
    }
}



__kernel void downfilter_y_g(
    __global uchar4* src,
    __global uchar4 *dst, int w, int h )
{

    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    float x0 = src[2*ix+(2*iy-2)*w*2].x/16.0;
    float x1 = src[2*ix+(2*iy-1)*w*2].x/8.0;
    float x2 = 3*src[2*ix+(2*iy)*w*2].x/4.0;
    float x3 = src[2*ix+(2*iy+1)*w*2].x/8.0;
    float x4 = src[2*ix+(2*iy+2)*w*2].x/16.0;
    
    int output = round(x0 + x1 + x2 + x3 + x4);

	if(output >255)
		output = 255;
    if( ix < w-2 && iy < h-2 ) {
        dst[iy*w + ix ] = (uchar4)(output,output,output,255);
    }
 
}