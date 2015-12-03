#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include "processor.h"

#include "string.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <CL/cl.hpp>

#include <android/bitmap.h>
#include <stdlib.h>

#include <jni.h>

cl::Context      gContext;
cl::CommandQueue gQueue;
cl::Kernel       gNV21Kernel;
cl::Kernel       gDownSamplerXKernel;
cl::Kernel       gDownSamplerYKernel;
cl::Buffer 		*bufferIn;
cl::Buffer 		*bufferOut;
cl::Buffer 		*bufferOut2;

const char* oclErrorString(cl_int error){
	static const char* errorString[] = {
			"CL_SUCCESS",
			"CL_DEVICE_NOT_FOUND",
			"CL_DEVICE_NOT_AVAILABLE",
			"CL_COMPILER_NOT_AVAILABLE",
			"CL_MEM_OBJECT_ALLOCATION_FAILURE",
			"CL_OUT_OF_RESOURCES",
			"CL_OUT_OF_HOST_MEMORY",
			"CL_PROFILING_INFO_NOT_AVAILABLE",
			"CL_MEM_COPY_OVERLAP",
			"CL_IMAGE_FORMAT_MISMATCH",
			"CL_IMAGE_FORMAT_NOT_SUPPORTED",
			"CL_BUILD_PROGRAM_FAILURE",
			"CL_MAP_FAILURE",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"CL_INVALID_VALUE",
			"CL_INVALID_DEVICE_TYPE",
			"CL_INVALID_PLATFORM",
			"CL_INVALID_DEVICE",
			"CL_INVALID_CONTEXT",
			"CL_INVALID_QUEUE_PROPERTIES",
			"CL_INVALID_COMMAND_QUEUE",
			"CL_INVALID_HOST_PTR",
			"CL_INVALID_MEM_OBJECT",
			"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
			"CL_INVALID_IMAGE_SIZE",
			"CL_INVALID_SAMPLER",
			"CL_INVALID_BINARY",
			"CL_INVALID_BUILD_OPTIONS",
			"CL_INVALID_PROGRAM",
			"CL_INVALID_PROGRAM_EXECUTABLE",
			"CL_INVALID_KERNEL_NAME",
			"CL_INVALID_KERNEL_DEFINITION",
			"CL_INVALID_KERNEL",
			"CL_INVALID_ARG_INDEX",
			"CL_INVALID_ARG_VALUE",
			"CL_INVALID_ARG_SIZE",
			"CL_INVALID_KERNEL_ARGS",
			"CL_INVALID_WORK_DIMENSION",
			"CL_INVALID_WORK_GROUP_SIZE",
			"CL_INVALID_WORK_ITEM_SIZE",
			"CL_INVALID_GLOBAL_OFFSET",
			"CL_INVALID_EVENT_WAIT_LIST",
			"CL_INVALID_EVENT",
			"CL_INVALID_OPERATION",
			"CL_INVALID_GL_OBJECT",
			"CL_INVALID_BUFFER_SIZE",
			"CL_INVALID_MIP_LEVEL",
			"CL_INVALID_GLOBAL_WORK_SIZE",
	};

	const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

	const int index = -error;

	return (index >= 0 && index < errorCount) ? errorString[index] : "Unspecified Error";
}

char *file_contents(const char *filename, int *length)
{
	FILE *f = fopen(filename, "r");
	void *buffer;

	if (!f) {
		LOGE("Unable to open %s for reading\n", filename);
		return NULL;
	}

	fseek(f, 0, SEEK_END);
	*length = ftell(f);
	fseek(f, 0, SEEK_SET);

	buffer = malloc(*length+1);
	*length = fread(buffer, 1, *length, f);
	fclose(f);
	((char*)buffer)[*length] = '\0';

	return (char*)buffer;
}

bool throwJavaException(JNIEnv *env,std::string method_name,std::string exception_msg, int errorCode=0)
{
	char buf[8];
	sprintf(buf,"%d",errorCode);
	std::string code(buf);

	std::string msg = "@" + method_name + ": " + exception_msg + " ";
	if(errorCode!=0) msg += code;

	jclass generalExp = env->FindClass("java/lang/Exception");
	if (generalExp != 0) {
		env->ThrowNew(generalExp, msg.c_str());
		return true;
	}
	return false;
}

void cb(cl_program p,void* data)
{
	clRetainProgram(p);
	cl_device_id devid[1];
	clGetProgramInfo(p,CL_PROGRAM_DEVICES,sizeof(cl_device_id),(void*)devid,NULL);
	char bug[65536];
	clGetProgramBuildInfo(p,devid[0],CL_PROGRAM_BUILD_LOG,65536*sizeof(char),bug,NULL);
//	clReleaseProgram(p);
	LOGE("Build log \n %s\n",bug);

	cl_build_status  status;
	clGetProgramBuildInfo(p,devid[0],CL_PROGRAM_BUILD_STATUS , sizeof(cl_build_status) ,&status ,NULL);
	LOGE("Build status \n %d\n",status );

	char option[65536];
	clGetProgramBuildInfo(p,devid[0],CL_PROGRAM_BUILD_OPTIONS , 65536*sizeof(char) ,&option ,NULL);
	LOGE("Build option \n %s\n",option );

	clReleaseProgram(p);

}



void process(uint32_t* outL1,uint32_t* outL2,uint32_t* outL3,  uint8_t* in, int w, int h)
{
	static int first = 1;
	cl::size_t<3> src_origin;
	cl::size_t<3> dst_origin;
	cl::size_t<3> region;
	cl::Event waitevent;

	dst_origin[0] = 0;
	dst_origin[1] = 0;
	dst_origin[2] = 0;

	src_origin[0] = 0;
	src_origin[1] = 0;
	src_origin[2] = 0;

	int nbthreadx = 16;
	int nbthready = 16;

	try {
		//LOGI("@process \n");
		gQueue.enqueueWriteBuffer( *bufferIn,CL_TRUE,0,w*h*sizeof(cl_uchar4),in);
		////////////////////////////////////
		// Conversion YUV to gray
		gNV21Kernel.setArg(0,*bufferOut);
		gNV21Kernel.setArg(1,*bufferIn);
		gNV21Kernel.setArg(2,w);
		gNV21Kernel.setArg(3,h);
		gQueue.enqueueNDRangeKernel(gNV21Kernel,
				cl::NullRange,
				cl::NDRange( (int)ceil((float)w/16.0f)*16,(int)ceil((float)h/16.0f)*16),
				cl::NDRange(16,16),
				NULL,
				NULL);

		//LOGI("@process1 \n");
		/////////////////////////////////
//		 Level 1

		gDownSamplerXKernel.setArg(0,*bufferOut);
		gDownSamplerXKernel.setArg(1,*bufferOut2);
		gDownSamplerXKernel.setArg(2,w);
		gDownSamplerXKernel.setArg(3,h);
		gQueue.enqueueNDRangeKernel(gDownSamplerXKernel,
				cl::NullRange,
				cl::NDRange( (int)ceil((float)w/nbthreadx)*nbthreadx,(int)ceil((float)h/nbthready)*nbthready),
				cl::NDRange(nbthreadx,nbthready),
				NULL,
				NULL);

		//LOGI("@process2 \n");
		gDownSamplerYKernel.setArg(0,*bufferOut2);
		gDownSamplerYKernel.setArg(1,*bufferOut);
		gDownSamplerYKernel.setArg(2,w/2);
		gDownSamplerYKernel.setArg(3,h/2);
		gQueue.enqueueNDRangeKernel(gDownSamplerYKernel,
				cl::NullRange,
				cl::NDRange( (int)ceil((float)w/2/nbthreadx)*nbthreadx,(int)ceil((float)h/2/nbthready)*nbthready),
				cl::NDRange(nbthreadx,nbthready),
				NULL,
				NULL);

		//LOGI("@process3 \n");
		//gQueue.enqueueReadBuffer( *bufferOut,CL_TRUE,0,w/2*h/2*sizeof(cl_uchar),outL1);
		//LOGI("@process4 \n");

	}
	catch (cl::Error e) {
		LOGI("@oclDecoder1: %s %d \n",e.what(),e.err());
	}

	try{
		/////////////////////////////////////////////
		// Level 2
		gDownSamplerXKernel.setArg(0,*bufferOut);
		gDownSamplerXKernel.setArg(1,*bufferOut2);
		gDownSamplerXKernel.setArg(2,w/2);
		gDownSamplerXKernel.setArg(3,h/2);
		gQueue.enqueueNDRangeKernel(gDownSamplerXKernel,
				cl::NullRange,
				cl::NDRange( (int)ceil((float)w/2/nbthreadx)*nbthreadx,(int)ceil((float)h/2/nbthready)*nbthready),
				cl::NDRange(nbthreadx,nbthready),
				NULL,
				NULL);

		gDownSamplerYKernel.setArg(0,*bufferOut2);
		gDownSamplerYKernel.setArg(1,*bufferOut);
		gDownSamplerYKernel.setArg(2,w/2);
		gDownSamplerYKernel.setArg(3,h/2);
		gQueue.enqueueNDRangeKernel(gDownSamplerYKernel,
				cl::NullRange,
				cl::NDRange( (int)ceil((float)w/2/nbthreadx)*nbthreadx,(int)ceil((float)h/2/nbthready)*nbthready),
				cl::NDRange(nbthreadx,nbthready),
				NULL,
				NULL);
		//gQueue.enqueueReadBuffer( *bufferOut ,CL_TRUE, 0, w/4*h/4*sizeof(cl_uchar),outL2);
	}
	catch (cl::Error e) {
		LOGI("Level 2: %s %d \n",e.what(),e.err());
	}




	try{
		/////////////////////////////////////
		// Level 3
		gDownSamplerXKernel.setArg(0,*bufferOut);
		gDownSamplerXKernel.setArg(1,*bufferOut2);
		gDownSamplerXKernel.setArg(2,w/4);
		gDownSamplerXKernel.setArg(3,h/4);
		gQueue.enqueueNDRangeKernel(gDownSamplerXKernel,
				cl::NullRange,
				cl::NDRange( (int)ceil((float)w/4/nbthreadx)*nbthreadx,(int)ceil((float)h/4/nbthready)*nbthready),
				cl::NDRange(nbthreadx,nbthready),
				NULL,
				NULL);

		gDownSamplerYKernel.setArg(0,*bufferOut2);
		gDownSamplerYKernel.setArg(1,*bufferOut);
		gDownSamplerYKernel.setArg(2,w/4);
		gDownSamplerYKernel.setArg(3,h/4);
		gQueue.enqueueNDRangeKernel(gDownSamplerYKernel,
				cl::NullRange,
				cl::NDRange( (int)ceil((float)w/4/nbthreadx)*nbthreadx,(int)ceil((float)h/4/nbthready)*nbthready),
				cl::NDRange(nbthreadx,nbthready),
				NULL,
				NULL);

		//gQueue.enqueueReadBuffer( *bufferOut ,CL_TRUE, 0, w/8*h/8*sizeof(cl_uchar),outL3);

	}
	catch (cl::Error e) {
		LOGI("Level 3: %s %d \n",e.what(),e.err());
	}

}



JNIEXPORT void JNICALL runfilter(
		JNIEnv *env,
		jclass clazz,
		jobject L1Bmp,
		jobject L2Bmp,
		jobject L3Bmp,
		jbyteArray inData,
		jint width,
		jint height)
{
	static int singleton =0;

	AndroidBitmapInfo bmpInfoL1;
	AndroidBitmapInfo bmpInfoL2;
	AndroidBitmapInfo bmpInfoL3;

	cl::ImageFormat image_format;
	image_format.image_channel_data_type	= CL_UNSIGNED_INT8;
	image_format.image_channel_order 		= CL_RGBA;


	if (AndroidBitmap_getInfo(env, L1Bmp, &bmpInfoL1) < 0) {
		throwJavaException(env,"gaussianBlur","Error retrieving bitmap meta data");
		return;
	}

	if (AndroidBitmap_getInfo(env, L2Bmp, &bmpInfoL2) < 0) {
		throwJavaException(env,"gaussianBlur","Error retrieving bitmap meta data");
		return;
	}

	if (AndroidBitmap_getInfo(env, L3Bmp, &bmpInfoL3) < 0) {
		throwJavaException(env,"gaussianBlur","Error retrieving bitmap meta data");
		return;
	}

//	if (bmpInfoL1.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
//		throwJavaException(env,"gaussianBlur","Expecting RGBA_8888 format");
//		return;
//	}
//
//	if (bmpInfoL2.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
//		throwJavaException(env,"gaussianBlur","Expecting RGBA_8888 format");
//		return;
//	}
//
//	if (bmpInfoL3.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
//		throwJavaException(env,"gaussianBlur","Expecting RGBA_8888 format");
//		return;
//	}


	uint32_t* bmpContentL1;
	uint32_t* bmpContentL2;
	uint32_t* bmpContentL3;

	if (AndroidBitmap_lockPixels(env, L1Bmp,(void**)&bmpContentL1) < 0) {
		throwJavaException(env,"gaussianBlur","Unable to lock bitmap pixels");
		return;
	}
	if (AndroidBitmap_lockPixels(env, L2Bmp,(void**)&bmpContentL2) < 0) {
		throwJavaException(env,"gaussianBlur","Unable to lock bitmap pixels");
		return;
	}
	if (AndroidBitmap_lockPixels(env, L3Bmp,(void**)&bmpContentL3) < 0) {
		throwJavaException(env,"gaussianBlur","Unable to lock bitmap pixels");
		return;
	}


	jbyte* inPtr = (jbyte*)env->GetPrimitiveArrayCritical(inData, 0);
	if (inPtr == NULL) {
		throwJavaException(env,"gaussianBlur","NV21 byte stream getPointer returned NULL");
		return;
	}

	LOGW("runfilter %d %d \n",width,height);
	process(bmpContentL1,bmpContentL2,bmpContentL3,(uint8_t*)inPtr,width,height);

	// This is absolutely neccessary before calling any other JNI function
	env->ReleasePrimitiveArrayCritical(inData,inPtr,0);
	AndroidBitmap_unlockPixels(env, L1Bmp);
	AndroidBitmap_unlockPixels(env, L2Bmp);
	AndroidBitmap_unlockPixels(env, L3Bmp);
}



JNIEXPORT jboolean JNICALL compileKernels(JNIEnv *env, jclass clazz,int w, int h)
{
	// Find OCL devices and compile kernels

	cl_int err = CL_SUCCESS;

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		return false;
	}

	cl_context_properties properties[] =	{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
	gContext = cl::Context(CL_DEVICE_TYPE_GPU, properties);
	std::vector<cl::Device> devices = gContext.getInfo<CL_CONTEXT_DEVICES>();
	gQueue = cl::CommandQueue(gContext, devices[1], 0, &err);

	LOGI("@compileKernels \n");
	int src_length = 0;
	const char* src  = file_contents("/data/data/com.example.subsamplecamera/app_execdir/kernels.cl",&src_length);
	cl::Program::Sources sources(1,std::make_pair(src, src_length+1));
	cl::Program *program;


	try {



		program = new cl::Program(gContext, sources,&err);
		LOGI("@compileKernels1.5 \n");
		program->build(devices,NULL,cb);
		while(program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[1]) != CL_BUILD_SUCCESS);
		gNV21Kernel 		= cl::Kernel(*program, "nv21togray", &err);
		gDownSamplerXKernel = cl::Kernel(*program, "downfilter_x_g", &err);
		gDownSamplerYKernel = cl::Kernel(*program, "downfilter_y_g", &err);

		bufferOut 	= new cl::Buffer(gContext, CL_MEM_READ_WRITE, w*h*sizeof(cl_uchar4));
		bufferOut2	= new cl::Buffer(gContext, CL_MEM_READ_WRITE, w*h*sizeof(cl_uchar4));
		bufferIn 	= new cl::Buffer(gContext, CL_MEM_READ_WRITE ,w*h*sizeof(cl_uchar4));

		return true;
	}
	catch (cl::Error e) {

		LOGI("@decode1 Build Status:: %s \n",program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]));
//		LOGI("@decode2:Build Options: %s \n", program->getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]));
//		LOGI("@decode3:Build Log:: %s \n",program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]));

		if( !throwJavaException(env,"decode",e.what(),e.err()) )
		{

//			LOGI("@decode0: %s \n",e.what());
//			LOGI("@decode1 Build Status:: %s \n",program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]));
//			LOGI("@decode2:Build Options: %s \n", program->getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]));
//			LOGI("@decode3:Build Log:: %s \n",program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]));

			return false;
		}
	}
}

JNIEXPORT jint JNICALL getSpeed(JNIEnv *env, jclass clazz)
{
	return 0;
}


static JNINativeMethod gMethodRegistryLiveFeatureActivity[] = {
		{ "compileKernels", "(II)Z", (void *) compileKernels },
		{ "runfilter", "(Landroid/graphics/Bitmap;Landroid/graphics/Bitmap;Landroid/graphics/Bitmap;[BII)V", (void *) runfilter },
		{ "getSpeed", "()I", (void *) getSpeed }
};


JNIEXPORT jint JNI_OnLoad(JavaVM* pVM, void* reserved) {
	JNIEnv *env;

	LOGW("JNI_OnLoad\n");
	if ((pVM->GetEnv((void **)(&env), JNI_VERSION_1_6)) != JNI_OK)
	{ //abort();
	}

	LOGW("JNI_OnLoad\n");
	jclass LiveFeatureActivity = env->FindClass("com/example/subsamplecamera/MainActivity");
	if (LiveFeatureActivity == NULL) abort();
	env->RegisterNatives( LiveFeatureActivity,gMethodRegistryLiveFeatureActivity, 3);
	env->DeleteLocalRef( LiveFeatureActivity);

	return JNI_VERSION_1_6;
}

