package com.example.subsamplecamera;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;

import android.app.Activity;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.TextureView;
import android.widget.FrameLayout;
import android.view.TextureView;
import android.widget.ImageView;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.graphics.SurfaceTexture;


public class MainActivity extends Activity implements
TextureView.SurfaceTextureListener, Camera.PreviewCallback {

	static {
        System.loadLibrary("JNIProcessor");
    }

    private final String TAG="LiveFeature";
	
    private Camera mCamera;
    private TextureView mTextureView;
    private byte[] mVideoSource;
	private ImageView mImageViewL1, mImageViewL2, mImageViewL3;
	private Bitmap mImageL1, mImageL2, mImageL3;

	    
	native private boolean compileKernels();
	native private void runfilter(Bitmap L1,Bitmap L2,Bitmap L3, byte[] in, int width, int height);


    private void copyFile(final String f) {
		InputStream in;
		try {
			in = getAssets().open(f);
			final File of = new File(getDir("execdir",MODE_PRIVATE), f);

			final OutputStream out = new FileOutputStream(of);

			final byte b[] = new byte[65535];
			int sz = 0;
			while ((sz = in.read(b)) > 0) {
				out.write(b, 0, sz);
			}
			in.close();
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
//		Log.i(TAG,"on create set contentviewOK");
		setContentView(R.layout.activity_main);
		
//        Log.i(TAG,"oncreateOK");
        copyFile("kernels.cl");
//        Log.i(TAG,"oncreate1OK");
        if( compileKernels() == false )
            Log.i(TAG,"Kernel compilation failed");

//        Log.i(TAG,"oncreate2ok");
        
//        Log.i(TAG,"oncreate1.1okok");
        mImageViewL1 = ((ImageView)findViewById(R.id.imageViewL1));
//        Log.i(TAG,"oncreate1.2");
        mImageViewL2 = ((ImageView)findViewById(R.id.imageViewL2));
//        Log.i(TAG,"oncreate1.3");
        mImageViewL3 = ((ImageView)findViewById(R.id.imageViewL3));
        mTextureView = (TextureView) findViewById(R.id.preview);

        
//        Log.i(TAG,"on create 2");
        
        mTextureView.setSurfaceTextureListener(this);

//        Log.i(TAG,"on create 3");  	
	}

	@Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture pSurface,
                                            int pWidth, int pHeight) {
        // Ignored
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture pSurface) {
        // Ignored
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture pSurface,
                                          int pWidth, int pHeight) {
        mCamera = Camera.open();
        try {
            mCamera.setPreviewTexture(pSurface);
            mCamera.setPreviewCallbackWithBuffer(this);
            // Sets landscape mode to avoid complications related to
            // screen orientation handling.
            mCamera.setDisplayOrientation(0);
            
            
            // Finds a suitable resolution.
            Size size = findBestResolution(pWidth, pHeight);
            PixelFormat pixelFormat = new PixelFormat();
            PixelFormat.getPixelFormatInfo(mCamera.getParameters()
                            .getPreviewFormat(), pixelFormat);
            int sourceSize = size.width * size.height
                            * pixelFormat.bitsPerPixel / 8;
            // Set-up camera size and video format. YCbCr_420_SP
            // should be the default on Android anyway.
            Camera.Parameters parameters = mCamera.getParameters();
            parameters.setPreviewSize(size.width, size.height);
            parameters.setPreviewFormat(PixelFormat.YCbCr_420_SP);
            mCamera.setParameters(parameters);

            
            // Prepares video buffer and bitmap buffers.
            mVideoSource = new byte[sourceSize];
            mImageL1 = Bitmap.createBitmap(size.width/2, size.height/2,
                                          Bitmap.Config.ARGB_8888);
            mImageL2 = Bitmap.createBitmap(size.width/4, size.height/4,
                                          Bitmap.Config.ARGB_8888);
            mImageL3 = Bitmap.createBitmap(size.width/8, size.height/8,
                                          Bitmap.Config.ARGB_8888);
            mImageViewL1.setImageBitmap(mImageL1);
            mImageViewL2.setImageBitmap(mImageL2);
            mImageViewL3.setImageBitmap(mImageL3);

            // Starts receiving pictures from the camera.
            mCamera.addCallbackBuffer(mVideoSource);
            mCamera.startPreview();
        } catch (IOException ioe) {
            mCamera.release();
            mCamera = null;
            throw new IllegalStateException();
        }
    }

    private Size findBestResolution(int pWidth, int pHeight) {
        List<Size> sizes = mCamera.getParameters()
                        .getSupportedPreviewSizes();
        // Finds the biggest resolution which fits the screen.
        // Else, returns the first resolution found.
        Size selectedSize = mCamera.new Size(0, 0);
        for (Size size : sizes) {
            if ((size.width <= pWidth)
             && (size.height <= pHeight)
             && (size.width >= selectedSize.width)
             && (size.height >= selectedSize.height)) {
                selectedSize = size;
            }
        }
        // Previous code assume that there is a preview size smaller
        // than screen size. If not, hopefully the Android API
        // guarantees that at least one preview size is available.
        if ((selectedSize.width == 0) || (selectedSize.height == 0)) {
            selectedSize = sizes.get(0);
        }
        return sizes.get(0);//selectedSize;
    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture pSurface)
    {
        // Releases camera which is a shared resource.
        if (mCamera != null) {
            mCamera.stopPreview();
            mCamera.release();
            // These variables can take a lot of memory. Get rid of
            // them as fast as we can.
            mCamera = null;
            mVideoSource = null;
            mImageL1.recycle(); mImageL1 = null;
            mImageL2.recycle(); mImageL2 = null;
            mImageL3.recycle(); mImageL3 = null;
        }
        return true;
    }
    
    @Override
    public void onPreviewFrame(byte[] pData, Camera pCamera) {
        // New data has been received from camera. Processes it and
        // requests surface to be redrawn right after.
        if (mCamera != null) {
            

        	runfilter(mImageL1,mImageL2,mImageL3, pData, mImageL1.getWidth()*2,mImageL1.getHeight()*2);
            mImageViewL1.invalidate();
            mImageViewL2.invalidate();
            mImageViewL3.invalidate();
            mCamera.addCallbackBuffer(mVideoSource);
        }
    }



}