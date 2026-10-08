#include "win32_ExpertSystems.h"

using namespace Eigen;
using namespace std;
#define WINWIDTH 400
#define WINHEIGHT 400
#define WINHALFWIDTH int(WINWIDTH * 0.5f)
#define WINHALFHEIGHT int(WINHEIGHT * 0.5f)
#define SCALE 100
global_variable bool globalRunning = true;
global_variable bool discreteOutput = false;
global_variable bool plotData = false;
global_variable bool profile = false;
global_variable int winTitleHeight = 10;
static clock_t startTime;
Buffer backBuffer;
Net neural;
Net d_neural;
NetTrainer h_trainer;
d_NetTrainer *d_trainer = nullptr;
global_variable float GraphZoom = 1.f;
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
		case 'D':
			discreteOutput = !discreteOutput;
			break;
		case 'W':
			h_trainer.ModifyLearningRate(0.02f);
			d_trainer->ModifyLearningRate(0.02f);
			break;
		case 'S':
			h_trainer.ModifyLearningRate(-0.02f);
			d_trainer->ModifyLearningRate(-0.02f);
			break;
		case 'Q':
			h_trainer.ModifyRegTerm(0.02f);
			d_trainer->ModifyRegTerm(0.02f);
			break;
		case 'A':
			h_trainer.ModifyRegTerm(-0.02f);
			d_trainer->ModifyRegTerm(-0.02f);
			break;
		case 'V':
			plotData = !plotData;
			break;
		case 'P':
			profile = !profile;
			break;
		default:
			break;
		}
	} break;
	case WM_MBUTTONDOWN:
	{
		GraphZoom = 1.f;
	}
	case WM_MOUSEWHEEL:
	{
		if (((short)HIWORD(WParam)) / 120 > 0) {
			PostMessage(Window, WM_VSCROLL, SB_LINEUP, (LPARAM)0);
			GraphZoom += 0.5f;
		}
		if (((short)HIWORD(WParam)) / 120 < 0) {
			PostMessage(Window, WM_VSCROLL, SB_LINEDOWN, (LPARAM)0);
			GraphZoom *= 0.5f;
		}
	}
	break;
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
void PlotData(MatrixXf X, MatrixXf Y) {
	for (int i = 0; i < X.cols(); ++i) {
		DrawFilledCircle(backBuffer, int(WINHALFWIDTH + X(0, i) * SCALE), int(WINHALFHEIGHT + -X(1, i) * SCALE), SCALE * 0.11f, Color(255, 255, 255, 255));
		DrawFilledCircle(backBuffer, int(WINHALFWIDTH + X(0, i) * SCALE), int(WINHALFHEIGHT + -X(1, i) * SCALE), SCALE * 0.08f,
			(Y(0, i) > 0.f ? positiveColor : negativeColor) - Color(50, 50, 50, 50));
	}
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
MatrixXf BuildPolynomials(MatrixXf m) {
	MatrixXf temp = MatrixXf(m.rows() + 3, m.cols());
	temp << m, MatrixXf(m.array().pow(2)), MatrixXf(m.array().pow(2)).colwise().sum();
	return temp;
}
void DrawOutputToScreen(const MatrixXf &screenCoords) {
	MatrixXf h = neural.ForwardPropagation(screenCoords);
	int *pixel = (int *)backBuffer.memory;
	for (int i = 0; i < h.cols(); ++i) {
		float percent = (*(h.data() + i));
		Color blended = Color(0, 0, 0, 0);
		switch (neural.GetParams().layerActivations.back()) {
		case Sigmoid:
			percent = (percent - 0.5f) * 2.f;
			break;
		case Linear:
		case Tanh:
		case ReLU:
		case LReLU:
		case Sine:
		default:
			break;
		}
		if (discreteOutput) {
			blended = percent < 0.f ? negativeColor : positiveColor;
		}
		else {
			blended = percent < 0.0f ? Color(255, 255, 255, 255).Blend(negativeColor, -percent)
				: Color(255, 255, 255, 255).Blend(positiveColor, percent);
		}
		*pixel++ = blended.ToBit();
	}
}
void UpdateDisplay(MatrixXf X, MatrixXf Y, vector<float> &h_history, vector<float> &d_history, const MatrixXf& screenCoords) {
	if (globalRunning) {
		d_trainer->Visualization((int *)backBuffer.memory, backBuffer.width, backBuffer.height, discreteOutput);
		//DrawOutputToScreen(screenCoords);
		if (plotData) {
			PlotData(X, Y);
		}
		//DrawHistory(backBuffer, h_history, Color(200, 100, 100, 255));
		DrawHistory(backBuffer, d_history, Color(100, 100, 200, 255));
	}
}
float totalTime;
void UpdateWinTitle(int &steps, HWND window) {
	const d_NetProfiler *profiler = d_trainer->GetProfiler();
	float d_time = profiler->forwardTime + profiler->backpropTime + profiler->updateTime + profiler->calcCostTime;// +profiler->visualizationTime;
	totalTime += d_time;
	float avgTime = totalTime / float(steps);
	char s[255];
	if (profile) {
		sprintf_s(s, "Total %0.5f | F %0.5f | B %0.5f | U %0.5f | C %0.5f | V %0.5f |"
			, d_time, profiler->forwardTime, profiler->backpropTime, profiler->updateTime, profiler->calcCostTime, profiler->visualizationTime);
	}
	else {
		sprintf_s(s, "SpiralData || Epoch %d | Time: %0.2f | milliseconds %0.10f | d_Cost %0.10f  "
			, steps++, ((float)(clock() - startTime)) / CLOCKS_PER_SEC, avgTime, d_trainer->GetCache().cost);
	}
	SetWindowText(window, LPCSTR(s));
}
int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode) {
	MatrixXf X;// = (MatrixXf)BuildMatFromFile("Spiral.csv"); write_binary("Spiral_64.dat", X);
	MatrixXf Y;// = (MatrixXf)BuildMatFromFile("SpiralLabels.csv"); write_binary("SpiralLabels_64.dat", Y);

			   //X = (MatrixXf)BuildMatFromFile("Spiral.csv"); write_binary("Spiral_64_float.dat", X);
			   //Y = (MatrixXf)BuildMatFromFile("SpiralLabels.csv"); write_binary("SpiralLabels_64_float.dat", Y);
			   //#if x64
			   //	read_binary("Spiral_64.dat", X);
			   //	read_binary("SpiralLabels_64.dat", Y);
			   //#else
			   //	read_binary("Spiral.dat", X);
			   //	read_binary("SpiralLabels.dat", Y);
			   //#endif
	read_binary("Spiral_64_float.dat", X);
	read_binary("SpiralLabels_64_float.dat", Y);
	X = BuildPolynomials(X);
	//X.conservativeResize(int(X.rows()), int(X.cols()*0.25f));
	//Y.conservativeResize(int(Y.rows()), int(Y.cols()*0.25f));
	WNDCLASSA winClass = {};
	InitializeWindow(&winClass, Instance, Win32MainWindowCallback, &backBuffer, WINWIDTH, WINHEIGHT, "NN_Spiral");
	const MatrixXf screenCoords = BuildPolynomials(BuildDisplayCoords(backBuffer, SCALE).transpose());
	if (RegisterClassA(&winClass)) {
		const HWND window = CreateWindowExA(0, winClass.lpszClassName, "NNet||",
		                                    WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MAXIMIZEBOX | WS_THICKFRAME | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
		                                    WINWIDTH * 2, WINHEIGHT * 2, 0, 0, Instance, 0);
		neural = Net((int)X.rows(), { 8, 8 }, (int)Y.rows(), {
			Tanh,
			Tanh,
			Tanh });
		//h_trainer = NetTrainer(&neural, X, Y, 1.f, 2.f, 20.f);
		if (!CanStartTraining(neural, X, Y)) {
			return EXIT_FAILURE;
		}
		d_neural = Net(neural);
		d_NetTrainer d_net_trainer(&d_neural, X, Y, 1.f, 2.f, 20.f, d_NetBatchParams(2,SlideWindow));
		d_trainer = &d_net_trainer;
		startTime = clock();
		const HDC deviceContext = GetDC(window);
		vector<float> h_history;
		vector<float> d_history;
		int steps = 0;
		d_trainer->BuildVisualization(screenCoords, (int *)backBuffer.memory, backBuffer.width, backBuffer.height);
		//Main Loop
		while ( /*steps < 1500 &&*/ globalRunning) {
			for (int epoch = 0; epoch < 1; ++epoch) {
				Win32ProcessPendingMessages();
				if (!globalRunning) {
					break;
				}
				//h_trainer.TrainSingleEpoch();
				d_trainer->TrainSingleEpoch();
				//UpdateHistory(h_history, h_trainer.GetCache().cost);
				UpdateHistory(d_history, d_trainer->GetCache().cost);
				UpdateWinTitle(steps, window);
			}
			UpdateDisplay(X, Y, h_history, d_history, screenCoords);
			Win32DisplayBufferInWindow(deviceContext, window, backBuffer);
		}
		DeleteDC(deviceContext);
	}
	return EXIT_SUCCESS;
}