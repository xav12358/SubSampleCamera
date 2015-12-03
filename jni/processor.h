#ifndef __JNI_H__
#define __JNI_H__
#ifdef __cplusplus

#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#define app_name "JNIProcessor"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, app_name, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, app_name, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, app_name, __VA_ARGS__))
extern "C" {
#endif

JNIEXPORT jint JNICALL getSpeed(JNIEnv *env, jclass clazz);
JNIEXPORT jboolean JNICALL compileKernels(JNIEnv *env, jclass clazz, int w,int h);
JNIEXPORT void JNICALL runfilter(
		JNIEnv *env,
		jclass clazz,
		jobject L1Bmp,
		jobject L2Bmp,
		jobject L3Bmp,
		jbyteArray inData,
		jint width,
		jint height);

#ifdef __cplusplus
}
#endif

#endif //__JNI_H__
