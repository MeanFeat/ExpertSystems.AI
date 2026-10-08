#include "win32_ExpertSystems.h"

using namespace Eigen;
using namespace std;
#define WINWIDTH 400
#define WINHEIGHT WINWIDTH
#define WINHALFWIDTH int((WINWIDTH-1)*0.5f)
#define WINHALFHEIGHT WINHALFWIDTH
#define PLOTDIST 90.f
#define WINDOWSTRETCHSCALE 2
global_variable bool globalRunning = true;
global_variable bool isTraining = true;
global_variable bool drawOutput = true;
global_variable bool plotData = false;
global_variable vector<float> predictions;
static clock_t startTime;
float mouseX = WINHALFWIDTH;
float mouseY = WINHALFHEIGHT + 100;
Buffer backBuffer;
Net neural;
d_NetTrainer *trainer = nullptr;
internal LRESULT CALLBACK Win32MainWindowCallback(HWND Window, UINT Message, WPARAM WParam, LPARAM LParam) {
	LRESULT Result = 0;
	switch (Message) {
	case WM_DESTROY:
	case WM_CLOSE:
	{
		globalRunning = false;
	} break;
	case WM_KEYDOWN:
	{
		switch (WParam) {
		case 'T':
			isTraining = !isTraining;
			break;
		case 'D':
			drawOutput = !drawOutput;
			break;
		case 'P':
			plotData = !plotData;
			break;
		case 'W':
			trainer->ModifyLearningRate(0.02f);
			break;
		case 'S':
			trainer->ModifyLearningRate(-0.02f);
			break;
		case 'Q':
			trainer->ModifyRegTerm(0.02f);
			break;
		case 'A':
			trainer->ModifyRegTerm(-0.02f);
			break;
		default:
			break;
		}
	} break;
	case WM_LBUTTONDOWN: //fall through
	case WM_MOUSEMOVE:
	{
		if (DWORD(WParam) & MK_LBUTTON) {
			POINT mousePos;
			GetCursorPos(&mousePos);
			ScreenToClient(Window, &mousePos);
			mouseX = float(mousePos.x / WINDOWSTRETCHSCALE);
			mouseY = float(mousePos.y / WINDOWSTRETCHSCALE);
		}
	} break;
	case WM_ACTIVATEAPP:
	{
		OutputDebugStringA("WM_ACTIVATEAPP\n");
	} break;
	default:
	{
		Result = DefWindowProcA(Window, Message, WParam, LParam);
	} break;
	}
	return Result;
}
MatrixXf NormalizeData(MatrixXf m) {
	MatrixXf a = (m.row(0).array().pow(2) + m.row(1).array().pow(2)).array().sqrt();
	MatrixXf b = MatrixXf(2, a.cols());
	b << a, a;
	return MatrixXf(m.array() / b.array());
}
Color GetColorBlend(float percent) {
	return percent < 0.f ? Color(255, 255, 255, 255).Blend(negativeColor, -percent)
		: Color(255, 255, 255, 255).Blend(positiveColor, percent);;
}
void PlotData(MatrixXf X, MatrixXf Y) {
	for (int i = 0; i < X.cols(); ++i) {
		DrawFilledCircle(backBuffer, int(WINHALFWIDTH + X(0, i)), int(WINHALFHEIGHT + -X(1, i)), 8.f, Color(0, 0, 0, 0));
		DrawFilledCircle(backBuffer, int(WINHALFWIDTH + X(0, i)), int(WINHALFHEIGHT + -X(1, i)), 6.f, GetColorBlend(Y(0, i)));
	}
}
void DrawOutputToScreen() {
	trainer->Visualization((int*)backBuffer.memory, backBuffer.width, backBuffer.height, false);
}
void UpdateDisplay(MatrixXf X, MatrixXf Y, vector<float> &history, vector<float> &testHistory, const MatrixXf& screenCoords) {
	if (globalRunning) {
		if (drawOutput) {
			DrawOutputToScreen();
		}
		else {
			ClearScreen(backBuffer);
		}
		if (plotData) {
			PlotData(X*PLOTDIST, Y);
		}
		DrawLine(backBuffer, WINHALFWIDTH, WINHALFHEIGHT, mouseX, float(WINHEIGHT - mouseY), Color(0, 0, 255, 255));
		for (int i = 0; i < (int)predictions.size(); ++i) {
			int predX = int(sin(predictions[i] * Pi32) * 100.f);
			int predY = int(cos(predictions[i] * Pi32) * 100.f);
			DrawLine(backBuffer, WINHALFWIDTH, WINHALFHEIGHT, float(WINHALFWIDTH) + predX, float(WINHALFHEIGHT) - predY, Color(100, 0, 0, 0));
		}
		DrawHistory(backBuffer, history, Color(200, 90, 90, 255));
		DrawHistory(backBuffer, testHistory, Color(100, 100, 100, 255));
	}
}
void UpdateWinTitle(int &steps, HWND window) {
	char s[255];
	sprintf_s(s, "%d|T:%0.2f|C:%0.10f|LR:%0.2f|RT:%0.2f|"
		, steps, ((float)(clock() - startTime)) / CLOCKS_PER_SEC,
		trainer->GetCache().cost, trainer->GetTrainParams().learnMult/trainer->GetTrainParams().learnCoeff, 
		trainer->GetTrainParams().regMod / trainer->GetTrainParams().learnCoeff);
	char r[255];
	sprintf_s(r, " |%0.2f|%0.2f| ", atan2((mouseX - WINHALFWIDTH), (mouseY - WINHALFHEIGHT)), predictions[0] * Pi32);
	strcat_s(s, r);
	SetWindowText(window, LPCSTR(s));
}
MatrixXf BuildRadians(MatrixXf m) {
	float coefficient = 1.f / Pi32;
	MatrixXf out = MatrixXf(1, m.cols());
	for (int i = 0; i < m.cols(); ++i) {
		out(0, i) = atan2((m(0, i)), (m(1, i))) * coefficient;
		//out(1, i) = atan2(-(m(1, i)), (m(0, i))) * coefficient;
		//out(2, i) = atan2((m(1, i)), -(m(0, i))) * coefficient;
	}
	return out;
}
void UpdatePrediction() {
	trainer->RefreshHostNetwork();
	MatrixXf mouse = MatrixXf(2, 1);
	mouse(0, 0) = mouseX - WINHALFWIDTH;
	mouse(1, 0) = mouseY - WINHALFHEIGHT;
	MatrixXf h = neural.ForwardPropagation(NormalizeData(mouse));
	predictions.clear();
	for (int i = 0; i < h.rows(); ++i) {
		predictions.push_back(h(i, 0));
	}
}
MatrixXf CreateSparseData(int pointCount) {
	float piAdjusted = Pi32 - 0.01f;
	float delta = (2.f * Pi32) / float(pointCount);
	MatrixXf out = MatrixXf(2, pointCount + 1);
	for (int i = 0; i <= pointCount; i++) {
		float r = clamp(-Pi32 + (delta * i), -piAdjusted, piAdjusted);
		out(0, i) = sin(r);
		out(1, i) = cos(r);
	}
	return out;
}
void UpdateHistory(vector<float> &hist, float cost) {
	float scale = (1.f - exp(-cost));
	hist.push_back(min((WINHEIGHT *  scale - cost) + backBuffer.titleOffset + 15, WINHEIGHT));
	if (hist.size() >= WINWIDTH + WINWIDTH) {
		for (int i = 1; i < (int)hist.size(); i += 2) {
			hist.erase(hist.begin() + i);
		}
	}
}
int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf X;
	MatrixXf testX;
	MatrixXf Y;
	MatrixXf testY;
	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance, Win32MainWindowCallback, &backBuffer, WINWIDTH, WINHEIGHT, "NN_PredictRadian");
	MatrixXf screenCoords = NormalizeData(BuildDisplayCoords(backBuffer).transpose());
	testX = NormalizeData(CreateSparseData(99));
	testY = BuildRadians(testX);
	X = NormalizeData(CreateSparseData(35));
	//X = NormalizeData(CreateRandomData(15));
	Y = BuildRadians(X);
	startTime = clock();
	if (RegisterClassA(&winClass)) {
		HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
			WS_OVERLAPPED | WS_SYSMENU | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
			WINWIDTH * WINDOWSTRETCHSCALE, WINHEIGHT * WINDOWSTRETCHSCALE, 0, 0, Instance, 0);
		neural = Net((int)X.rows(), { 32,16 }, (int)Y.rows(), { Tanh,Tanh,Tanh });
		if (!CanStartTraining(neural, X, Y)) {
			return EXIT_FAILURE;
		}
		d_NetTrainer TempTrainer = d_NetTrainer(&neural, X, Y, 0.15f, 0.5f, 1.0f, d_NetBatchParams(2, SlideWindow));
		trainer = &TempTrainer;
		HDC deviceContext = GetDC(window);
		vector<float> history;
		vector<float> testHistory;
		initParallel();
		setNbThreads(4);
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		trainer->BuildVisualization(screenCoords, (int *)backBuffer.memory, backBuffer.width, backBuffer.height);
		int steps = 0;
		//Main Loop
		while (globalRunning) {
			Win32ProcessPendingMessages();
			if (isTraining) {
				trainer->TrainSingleEpoch();
				UpdateHistory(history, trainer->GetCache().cost);
				//UpdateHistory(testHistory, trainer.CalcCost(&neural.ForwardPropagation(testX), &testY));
				steps++;
			}
			else {
			}
			UpdateDisplay(X, Y, history, testHistory, screenCoords);
			UpdatePrediction();
			Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
			UpdateWinTitle(steps, window);
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}