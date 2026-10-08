#pragma  once
#if !defined(WIN32_EXPERTSYSTEMS_H)
#include "es_core_pch.h"
#include "stdDraw.h"
#include "stdMatrix.h"
#include "stdNet.h"
#include "stdNetTrainer.h"
#include "d_NetTrainer.h"

#define internal static
#define local_persist static
#define global_variable static

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef int32 bool32;

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef float real32;
typedef float real64;

#define Assert(Expression) if(!(Expression)) {*(int *)0 = 0;}
#define clamp(x,lo,hi) min( hi, max(lo,x) )

void InitializeWindow(WNDCLASSA *winclass, HINSTANCE instance, WNDPROC windowCallback, Buffer *backBuffer, int width, int height, LPSTR className) {
	backBuffer->bitmapInfo.bmiHeader.biSize = sizeof(backBuffer->bitmapInfo.bmiHeader);
	backBuffer->bitmapInfo.bmiHeader.biWidth = width;
	backBuffer->bitmapInfo.bmiHeader.biHeight = height;
	backBuffer->bitmapInfo.bmiHeader.biPlanes = 1;
	backBuffer->bitmapInfo.bmiHeader.biBitCount = 32;
	backBuffer->bitmapInfo.bmiHeader.biCompression = BI_RGB;
	backBuffer->memory = malloc(width * height * 4);
	backBuffer->width = width;
	backBuffer->height = height;
	backBuffer->titleOffset = 0;
	winclass->style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	winclass->lpfnWndProc = windowCallback;
	winclass->hInstance = instance;
	winclass->lpszClassName = className;
}

internal void Win32ProcessPendingMessages() {
	MSG Message;
	while (PeekMessage(&Message, 0, 0, 0, PM_REMOVE)) {
		TranslateMessage(&Message);
		DispatchMessageA(&Message);
	}
}

internal void Win32DisplayBufferInWindow(HDC DeviceContext, HWND hwind, Buffer buffer) {
	RECT winRect = {};
	GetWindowRect(hwind, &winRect);
	StretchDIBits(DeviceContext, 0, 0, winRect.right - winRect.left, winRect.bottom - winRect.top,
		0, 0, buffer.width, buffer.height, buffer.memory, &buffer.bitmapInfo, DIB_RGB_COLORS, SRCCOPY);
}

#define WIN32_EXPERTSYSTEMS_H
// Start guard: a valid network and train data are required, test data is optional.
inline bool CanStartTraining(Net &network, const Eigen::MatrixXf &trainData, const Eigen::MatrixXf &trainLabels) {
	const char *error = nullptr;
	if (network.GetNodeCount() <= 0) {
		error = "Cannot start: the network is invalid.";
	}
	else if (trainData.size() == 0 || trainLabels.size() == 0) {
		error = "Cannot start: train data is missing or empty.";
	}
	else if (trainData.cols() != trainLabels.cols()) {
		error = "Cannot start: train data and labels have different example counts.";
	}
	if (error) {
		MessageBoxA(nullptr, error, "ExpertSystems.AI", MB_OK | MB_ICONERROR);
		return false;
	}
	return true;
}

#endif