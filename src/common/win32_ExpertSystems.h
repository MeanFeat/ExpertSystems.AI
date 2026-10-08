#pragma  once
#if !defined(WIN32_EXPERTSYSTEMS_H)
#include "es_core_pch.h"
#include "stdDraw.h"
#include "stdMatrix.h"
#include "stdNet.h"
#include "stdNetTrainer.h"
#include "d_NetTrainer.h"
#include <commdlg.h>
#include <cstring>
#include <string>

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

internal std::string Win32SelectFile(HWND owner, const char *title, const char *filter, const char *initialPath, bool save) {
	char filePath[MAX_PATH] = {};
	if (initialPath) {
		strncpy_s(filePath, initialPath, _TRUNCATE);
	}
	OPENFILENAMEA dialog = {};
	dialog.lStructSize = sizeof(dialog);
	dialog.hwndOwner = owner;
	dialog.lpstrFilter = filter;
	dialog.lpstrFile = filePath;
	dialog.nMaxFile = sizeof(filePath);
	dialog.lpstrTitle = title;
	const char *extension = initialPath ? std::strrchr(initialPath, '.') : nullptr;
	dialog.lpstrDefExt = extension ? extension + 1 : nullptr;
	dialog.Flags = OFN_EXPLORER | OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST |
		(save ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);
	if (save ? GetSaveFileNameA(&dialog) : GetOpenFileNameA(&dialog)) {
		return filePath;
	}
	return std::string();
}

internal std::string Win32OpenFileDialog(HWND owner, const char *title, const char *filter, const char *initialPath = nullptr) {
	return Win32SelectFile(owner, title, filter, initialPath, false);
}

internal std::string Win32SaveFileDialog(HWND owner, const char *title, const char *filter, const char *initialPath = nullptr) {
	return Win32SelectFile(owner, title, filter, initialPath, true);
}

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
#endif