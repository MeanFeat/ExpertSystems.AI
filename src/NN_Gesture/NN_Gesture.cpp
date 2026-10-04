#include "win32_ExpertSystems.h"

using namespace Eigen;
using namespace std;
#define WINWIDTH 800
#define WINHEIGHT WINWIDTH
#define WINHALFWIDTH int((WINWIDTH-1)*0.5f)
#define WINHALFHEIGHT WINHALFWIDTH
#define MAXCAPTURECOUNT 25
#define MOUSETRAILLENGTH 300
#define CAPTURETHRESHOLD 40.f

global_variable bool globalRunning = true;
global_variable bool isTraining = true;
global_variable bool isVerifying = false;
global_variable bool isRecordingData = false;
global_variable bool shouldSaveChanges = false;
global_variable bool skipStep = false;
global_variable float successFade = 0.f;
global_variable float historyZoom = 5.f;
global_variable bool isMouseMoving = false;

global_variable vector<Vector2f> mouseCapture;
global_variable vector<Vector2f> lastSuccesful8;
global_variable vector<Vector2f> lastSuccesful8Deltas;
global_variable vector<Vector2f> deltaCapture;
global_variable vector<vector<Vector2f>> samples;
global_variable vector<vector<Vector2f>> deltas;
global_variable vector<float> labels;
global_variable bool isCapturingEight = false;

global_variable float *verify;
global_variable float *verifyLabel;

static clock_t startTime;
static float trainingTime;

float mouseX = WINHALFWIDTH;
float mouseY = WINHALFHEIGHT;

Net neural;
d_NetTrainer *trainer = nullptr;
Buffer backBuffer;

void RecordSample(vector<Vector2f> vec, float label) {
	vector<Vector2f> mods;
	mods.push_back(Vector2f(1.f, 1.f));
	mods.push_back(Vector2f(-1.f, 1.f));
	mods.push_back(Vector2f(1.f, -1.f));
	mods.push_back(Vector2f(-1.f, -1.f));
	vector<Vector2f> sample = mouseCapture;
	vector<Vector2f> delta = deltaCapture;
	for (int c = 0; c < mods.size(); c++) {
		vector<Vector2f> tempSample = sample;
		vector<Vector2f> tempDelta = delta;
		for (int i = 0; i < MAXCAPTURECOUNT; i++) {
			tempSample[i] = Vector2f(tempSample[i][0] * mods[c][0] + (mods[c][0] > 0 ? 0 : WINHEIGHT),
				tempSample[i][1] * mods[c][1] + (mods[c][1] > 0 ? 0 : WINWIDTH));
			tempDelta[i] = Vector2f(tempDelta[i][0] * mods[c][0], tempDelta[i][1] * mods[c][1]);
		}
		samples.push_back(tempSample);
		deltas.push_back(tempDelta);
		labels.push_back(label);
	}
}

void CopyNestedVec(MatrixXf &saveData, vector<vector<Vector2f>> * s, int maxCount) {
	int k = 0;
	for (int i = 0; i < saveData.cols(); i++) {
		for (int j = 0; j < maxCount; j++) {
			*(saveData.data() + k++) = float((*s)[i][j][0]);
			*(saveData.data() + k++) = float((*s)[i][j][1]);
		}
	}
}

internal LRESULT CALLBACK Win32MainWindowCallback(HWND Window, UINT Message, WPARAM WParam, LPARAM LParam) {
	LRESULT Result = 0;
	isMouseMoving = false;
	switch (Message) {
	case WM_DESTROY:
	case WM_CLOSE:
	{
		globalRunning = false;
	} break;
	case WM_MOUSEMOVE:
	{
		isMouseMoving = true;
		mouseX = float(GET_X_LPARAM(LParam));
		mouseY = float(GET_Y_LPARAM(LParam)) + backBuffer.titleOffset - 25;
	} //fall through
	case WM_KEYUP:
	{
		switch (WParam) {
		case '8':
			isCapturingEight = false;
			RecordSample(deltaCapture, 1.f);
			break;
		case 'V':
			*verifyLabel = 1.f;
			skipStep = true;
			break;
		case 'C':
			*verifyLabel = 0.f;
			skipStep = true;
			break;
		case 'T':
			if (isTraining) {
				trainer->RefreshHostNetwork();
			}
			isTraining = !isTraining;
			break;
		case '0':
			RecordSample(deltaCapture, 0.f);
			shouldSaveChanges = true;
			break;
		case 'S':
		{
			MatrixXf saveSamples = MatrixXf(MAXCAPTURECOUNT * 2, samples.size() - 1);
			CopyNestedVec(saveSamples, &samples, MAXCAPTURECOUNT);
			if (saveSamples.size() > MAXCAPTURECOUNT) {
				writeToCSVfile("Ideal8.csv", saveSamples.transpose());
			}
		}
		break;
		}
	} break;
	case WM_KEYDOWN:
	{
		switch (WParam) {
		case '8':
			isCapturingEight = true;
			break;
		default:
			break;
		}
	}
	case WM_MOUSEWHEEL:
	{
		if (((short)HIWORD(WParam)) / 120 > 0) {
			PostMessage(Window, WM_VSCROLL, SB_LINEUP, (LPARAM)0);
			historyZoom += 0.5f;
		}
		if (((short)HIWORD(WParam)) / 120 < 0) {
			PostMessage(Window, WM_VSCROLL, SB_LINEDOWN, (LPARAM)0);
			historyZoom = max(1.f, historyZoom - 0.5f);
		}
	}
	case WM_ACTIVATEAPP:
	{
		ShowCursor(false);
		OutputDebugStringA("WM_ACTIVATEAPP\n");
	} break;
	default: {
		Result = DefWindowProcA(Window, Message, WParam, LParam);
	} break;
	}
	return Result;
}

#define ICONLEFT 50
#define ICONTOP	100
#define HISTORYZOOM(a)((1.f - exp(-a) ) * historyZoom)
void UpdateDisplay(vector<Vector2f> &i8, vector<Vector2f> &mTrail, vector<Vector2f> &mCap, const vector<float> &hist, float h) {
	if (globalRunning) {
		ClearScreen(backBuffer);

		if (isTraining) {
			for (float i = 0.f; i <= 4.f; i += 0.1f) {
				float scale = HISTORYZOOM(i);
				float invScale = 1.f - scale;
				int height = min(int((WINHEIGHT * scale) + backBuffer.titleOffset + 15), WINHEIGHT);
				DrawLine(backBuffer, 0.f, float(height), float(WINWIDTH), float(height),
					Color(int(50 * invScale), int(50 * invScale), int(50 * invScale), 0));
			}
			vector<float> scaledHist = hist;
			for (auto& x : scaledHist) {
				x = min(x*historyZoom, WINHEIGHT);
			}
			DrawHistory(backBuffer, scaledHist, Color(200, 100, 100, 255));
		}
		else {
			for (int i = -3; i < 3; i++) {
				if (successFade > 0.f) {
					DrawLine(backBuffer, float(ICONLEFT), float(ICONTOP + i), float(ICONLEFT - 25), float(ICONTOP + i + 25), Color(0, int(255 * successFade), 0, 0));
					DrawLine(backBuffer, float(ICONLEFT), float(ICONTOP + i), float(ICONLEFT + 75), float(ICONTOP + i + 75), Color(0, int(255 * successFade), 0, 0));
				}
				else {
					DrawLine(backBuffer, float(ICONLEFT + 50), float(ICONTOP + i - 50), float(ICONLEFT - 50), float(ICONTOP + i + 50), Color(200, 0, 0, 0));
					DrawLine(backBuffer, float(ICONLEFT - 50), float(ICONTOP + i - 50), float(ICONLEFT + 50), float(ICONTOP + i + 50), Color(200, 0, 0, 0));
				}
				for (int mT = 1; mT < mTrail.size() - 1; mT++) {
					DrawLine(backBuffer, mTrail[mT - 1][0], float(WINHEIGHT) - mTrail[mT - 1][1],
						mTrail[mT][0], float(WINHEIGHT) - mTrail[mT][1], Color(200, 200, 200, 0) * (float(mT) / float(MOUSETRAILLENGTH)));
				}
				if (isRecordingData && mCap.size()) {
					for (int mC = 0; mC < mCap.size(); mC++) {
						DrawFilledCircle(backBuffer, int(mCap[mC][0]), WINHEIGHT - int(mCap[mC][1]), 10.f, isCapturingEight ? Color(0, 200, 0, 0) : Color(200, 200, 200, 0));
					}
				}
				if (h > 0.95f) {
					lastSuccesful8.clear();
					lastSuccesful8Deltas.clear();
					lastSuccesful8 = mouseCapture;
					lastSuccesful8Deltas = deltaCapture;
					successFade = 1.f;
				}
				if (successFade > 0.f && lastSuccesful8.size() > 0) {
					for (int ls = 0; ls < lastSuccesful8.size(); ls++) {
						DrawFilledCircle(backBuffer, int(lastSuccesful8[ls][0]), WINHEIGHT - int(lastSuccesful8[ls][1]), 10.f, Color(0, int(float(ls) * 10.f * successFade), 0, 0));
					}
				}

				if (i8.size() > 0) {
					Color base = Color(50, 50, 50, 0);
					Color blend = base.Blend(Color(150, 150, 150, 0), successFade);
					for (int e = 0; e < i8.size(); e++) {
						DrawFilledCircle(backBuffer, int(i8[e][0]), WINHEIGHT - int(i8[e][1]), 10.f, blend);
					}
				}
			}
			successFade = max(0.f, successFade - 0.001f);
			DrawFilledCircle(backBuffer, int(mouseX), WINHEIGHT - int(mouseY), 5.f, Color(200, 200, 200, 0));
		}
	}
}

void UpdateWinTitle(int &steps, float &prediciton, HWND window) {
	char s[255];
	if (isTraining) {
		trainingTime = ((float)(clock() - startTime)) / CLOCKS_PER_SEC;
		sprintf_s(s, "Epoch %d || Time: %0.2f || Eps %0.2f || Cost: %0.10f || ", steps, trainingTime, float(steps) / trainingTime, trainer->GetCache().cost);
	}
	else {
		sprintf_s(s, "TrainingTime: %0.1f || Cost: %0.10f || Prediction: %0.10f || ", trainingTime, trainer->GetCache().cost, prediciton);
		char prompt[255];
		if (successFade > 0.f) {
			sprintf_s(prompt, "Great!");
		}
		else {
			sprintf_s(prompt, "hmmm... Draw a figure eight.");
		}
		strcat_s(s, prompt);
	}
	SetWindowText(window, LPCSTR(s));
}

void CopyVecToMatrix(MatrixXf &dest, vector<vector<Vector2f>> orig) {
	int k = 0;
	for (int i = 0; i < orig.size() - 1; i++) {
		for (int j = 0; j < MAXCAPTURECOUNT; j++) {
			*(dest.data() + k++) = float(orig[i][j][0]);
			*(dest.data() + k++) = float(orig[i][j][1]);
		}
	}
}

void ContainVector(vector<Vector2f> &vec, int maxSize) {
	if (vec.size() > maxSize) {
		vec.erase(vec.begin());
	}
}

void UpdateHistory(vector<float> &hist) {
	float scale = (1.f - exp(-trainer->GetCache().cost));
	hist.push_back(min((WINHEIGHT *  scale - trainer->GetCache().cost) + backBuffer.titleOffset + 15, WINHEIGHT));
	if (hist.size() >= WINWIDTH + WINWIDTH) {
		for (int i = 1; i < (int)hist.size(); i += 2) {
			hist.erase(hist.begin() + i);
		}
	}
}

int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf readSamples;//= BuildMatFromFile("GestureSamplesTrain.csv").transpose();
	MatrixXf readDeltas;//= BuildMatFromFile("GroupedDeltas.csv").transpose();
	MatrixXf readLabels;//=BuildMatFromFile("GroupedLabels.csv").transpose();
	MatrixXf readIdeal8 = MatrixXf(50, 1);
	read_binary("GroupedDeltas_64.dat", readDeltas);
	read_binary("GroupedLabels_64.dat", readLabels);
	readIdeal8 << 355, 263, 397, 247, 437, 252, 471, 274, 490, 310, 492, 350, 470, 386, 440, 415, 407, 439, 374, 463, 347, 495, 342, 535, 349, 575, 374, 607, 414, 613, 454, 602, 479, 570, 486, 529, 480, 489, 451, 461, 414, 443, 382, 419, 353, 390, 340, 350, 332, 310;
	vector<Vector2f> ideal8;
	int i = 0;
	while (i < readIdeal8.rows()) {
		float xVal = (*(readIdeal8.data() + i++) - WINWIDTH) * 1.1f + WINWIDTH;
		float yVal = (*(readIdeal8.data() + i++) - WINWIDTH) * 1.1f + WINWIDTH;
		ideal8.push_back(Vector2f(xVal, yVal));
	}

	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance, Win32MainWindowCallback, &backBuffer, WINWIDTH, WINHEIGHT, "NN_Gesture");
	startTime = clock();
	if (RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
			WS_OVERLAPPED | WS_SYSMENU | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
			WINWIDTH, WINHEIGHT, 0, 0, Instance, 0);
		initParallel();
		setNbThreads(4);
		HDC deviceContext = GetDC(window);
		vector<Vector2f> mouseTrail;
		int sampleIndex = 0;
		//neural = Net("Gesture-Weights.json");
		neural = Net((int)readDeltas.rows(), { 100, 50 }, (int)readLabels.rows(), {
			Tanh,
			Tanh,
			Sigmoid });
		d_NetTrainer OrigTrainer(&neural, readDeltas, readLabels, 1.f, 1.25f, 0.0001f, d_NetBatchParams(10, SlideWindow));
		trainer = &OrigTrainer;
		vector<float> history;
		float h = 0.f;
		int steps = 0;
		//Main Loop
		while (globalRunning) {
			Win32ProcessPendingMessages();
			if (isTraining) {
				for (int e = 0; e < 1; e++) {
					trainer->TrainSingleEpoch();
				}
				UpdateHistory(history);
			}
			else if (isVerifying) {
				if (steps < readLabels.cols() && readLabels.cols() > 0) {
					verifyLabel = &readLabels(0, steps);
					if (!skipStep) {// && readLabels(0, steps) == 0.f){// && readLabels(1, steps) == 0.f ) {
						for (int i = 0; i < readSamples.rows(); ) {
							float valX = *(readSamples.data() + (steps * readSamples.rows() + i++));
							float valY = *(readSamples.data() + (steps * readSamples.rows() + i++));
							DrawFilledCircle(backBuffer, int(valX), WINHEIGHT - int(valY), 10.f, Color(i * 10, 200, 0, 0));
						}
					}
					else {
						skipStep = false;
						ClearScreen(backBuffer);
						steps += 4;
					}
					Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
					continue;
				}
				globalRunning = false;
			}
			else {
				Vector2f mousePos = Vector2f(float(mouseX), float(mouseY));
				if (mouseCapture.size() == 0) {
					mouseCapture.push_back(mousePos);
					mouseTrail.push_back(mousePos);
				}
				if (isMouseMoving) {
					ContainVector(mouseTrail, MOUSETRAILLENGTH);
					Vector2f mouseDelta = mouseTrail.back() - mousePos;
					if (sqrtf(mouseDelta[0] * mouseDelta[0] + mouseDelta[1] * mouseDelta[1]) >= 2.f) {
						mouseTrail.push_back(mousePos);
					}
					Vector2f delta = mouseCapture.back() - mousePos;
					if (sqrtf(delta[0] * delta[0] + delta[1] * delta[1]) >= CAPTURETHRESHOLD) {
						deltaCapture.push_back(delta / CAPTURETHRESHOLD);
						mouseCapture.push_back(mousePos);
						ContainVector(mouseCapture, MAXCAPTURECOUNT);
						ContainVector(deltaCapture, MAXCAPTURECOUNT);
						if (!isCapturingEight && deltaCapture.size() == MAXCAPTURECOUNT) {
							assert(deltaCapture.size() == mouseCapture.size());
							//Capture negative samples every MAXCAPTURECOUNT samples
							if (sampleIndex++ > MAXCAPTURECOUNT) {
								RecordSample(deltaCapture, 0.f);
								sampleIndex = 0;
							}
							if (!isRecordingData) {
								MatrixXf X = MatrixXf(MAXCAPTURECOUNT * 2, 1);
								int l = 0;
								for (int i = 0; i < deltaCapture.size(); i++) {
									float xVal = float(deltaCapture[i][0]);
									float yVal = float(deltaCapture[i][1]);
									*(X.data() + l++) = xVal;
									*(X.data() + l++) = yVal;
								}
								MatrixXf output = neural.ForwardPropagation(X);
								h = output(0, 0);
							}
						}
					}
				}
			}
			UpdateDisplay(ideal8, mouseTrail, mouseCapture, history, h);
			Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
			UpdateWinTitle(steps, h, window);
			steps++;
		}

		if ((!isTraining && isRecordingData) || shouldSaveChanges) {
			if (!isVerifying && samples.size() > MAXCAPTURECOUNT) {
				MatrixXf saveSamples = MatrixXf(MAXCAPTURECOUNT * 2, samples.size() - 1);
				MatrixXf saveDeltas = MatrixXf(MAXCAPTURECOUNT * 2, deltas.size() - 1);
				MatrixXf saveLabels = MatrixXf(1, samples.size() - 1);
				CopyNestedVec(saveSamples, &samples, MAXCAPTURECOUNT);
				CopyNestedVec(saveDeltas, &deltas, MAXCAPTURECOUNT);
				int l = 0;
				for (int i = 0; i < samples.size() - 1; i++) {
					*(saveLabels.data() + l++) = float(labels[i]);
				}
				if (readSamples.size() + readLabels.size() + readDeltas.size() > 0) {
					MatrixXf outSamples = MatrixXf(readSamples.rows(), readSamples.cols() + saveSamples.cols());
					outSamples << readSamples, saveSamples;
					MatrixXf outDeltas = MatrixXf(readDeltas.rows(), readDeltas.cols() + saveDeltas.cols());
					outSamples << readDeltas, saveDeltas;
					MatrixXf outLabels = MatrixXf(readLabels.rows(), readLabels.cols() + saveLabels.cols());
					outLabels << readLabels, saveLabels;
					writeToCSVfile("GestureSamplesTrain.csv", outSamples.transpose());
					writeToCSVfile("GestureDeltasTrain.csv", outDeltas.transpose());
					writeToCSVfile("GestureLabelsTrain.csv", outLabels.transpose());
				}
				else {
					writeToCSVfile("GestureSamplesTrain.csv", saveSamples.transpose());
					writeToCSVfile("GestureDeltasTrain.csv", saveDeltas.transpose());
					writeToCSVfile("GestureLabelsTrain.csv", saveLabels.transpose());
				}
			}
			else {
				writeToCSVfile("GestureLabelsTrain.csv", readLabels.transpose());
			}
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}