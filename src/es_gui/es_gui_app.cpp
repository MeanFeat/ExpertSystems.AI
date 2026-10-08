// Dear ImGui (Win32 + DirectX11) front end for training networks.
#include "es_core_pch.h"
#include "stdMatrix.h"
#include "es_async_trainer.h"
#include "es_profile.h"
#include <algorithm>
#include <d3d11.h>
#include <string>
#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx11.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

static ID3D11Device *g_dev;
static ID3D11DeviceContext *g_ctx;
static IDXGISwapChain *g_swap;
static ID3D11RenderTargetView *g_rtv;

static void CreateRTV() {
	ID3D11Texture2D *bb = nullptr;
	g_swap->GetBuffer(0, IID_PPV_ARGS(&bb));
	g_dev->CreateRenderTargetView(bb, nullptr, &g_rtv);
	bb->Release();
}
static void ReleaseRTV() { if (g_rtv) { g_rtv->Release(); g_rtv = nullptr; } }

static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
	if (ImGui_ImplWin32_WndProcHandler(h, m, w, l)) return true;
	switch (m) {
	case WM_SIZE:
		if (g_dev && w != SIZE_MINIMIZED) {
			ReleaseRTV();
			g_swap->ResizeBuffers(0, LOWORD(l), HIWORD(l), DXGI_FORMAT_UNKNOWN, 0);
			CreateRTV();
		}
		return 0;
	case WM_DESTROY: PostQuitMessage(0); return 0;
	}
	return DefWindowProc(h, m, w, l);
}

struct Buf { char s[260]; Buf(const char *d = "") { strncpy_s(s, d, _TRUNCATE); } };

static std::string DirOf(const std::string &p) {
	const size_t i = p.find_last_of("\\/");
	return i == std::string::npos ? "" : p.substr(0, i + 1);
}
static bool IsAbs(const std::string &p) { return p.size() > 1 && (p[1] == ':' || p[0] == '\\' || p[0] == '/'); }

static Eigen::MatrixXf LoadMatrix(const char *path) {
	const std::string filePath(path);
	Eigen::MatrixXf matrix;
	if (filePath.size() >= 4 && filePath.compare(filePath.size() - 4, 4, ".dat") == 0) {
		Eigen::read_binary(path, matrix);
	} else {
		matrix = Eigen::BuildMatFromFile(filePath);
	}
	return matrix;
}

int WINAPI WinMain(HINSTANCE inst, HINSTANCE, LPSTR, int) {
	WNDCLASSEX wc = { sizeof(wc), CS_CLASSDC, WndProc, 0, 0, inst, nullptr, nullptr, nullptr, nullptr, "esgui", nullptr };
	RegisterClassEx(&wc);
	HWND hwnd = CreateWindow(wc.lpszClassName, "ExpertSystems.AI Trainer", WS_OVERLAPPEDWINDOW, 100, 100, 1100, 700, nullptr, nullptr, inst, nullptr);

	DXGI_SWAP_CHAIN_DESC sd = {};
	sd.BufferCount = 2;
	sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.OutputWindow = hwnd;
	sd.SampleDesc.Count = 1;
	sd.Windowed = TRUE;
	sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	D3D_FEATURE_LEVEL fl;
	if (FAILED(D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &sd, &g_swap, &g_dev, &fl, &g_ctx))) return 1;
	CreateRTV();
	ShowWindow(hwnd, SW_SHOWDEFAULT);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui_ImplWin32_Init(hwnd);
	ImGui_ImplDX11_Init(g_dev, g_ctx);

	Net net;
	AsyncTrainer trainer;
	EsProfile profile;
	Buf profilePath("src\\es_gui\\gesture_train.esprofile"), netPath("net.json");
	Buf trainX, trainY, testX, testY;
	int inSize = 2, outSize = 1;
	std::vector<int> hiddenSizes{ 8, 8 };
	std::vector<Activation> hiddenActivations{ Tanh, Tanh };
	Activation outputActivation = Linear;
	const bool gpuAvailable = IsGpuAvailable();
	TrainerBackend backend = gpuAvailable ? TrainerBackend::Gpu : TrainerBackend::Cpu;
	std::string status = "Ready";

	auto SyncArchitecture = [&] {
		NetParameters &params = net.GetParams();
		const int depth = net.GetDepth();
		if (depth < 1 || params.layerSizes.size() != size_t(depth + 1) ||
			params.layerActivations.size() != size_t(depth)) {
			status = "Network architecture is invalid";
			return false;
		}
		inSize = params.layerSizes.front();
		outSize = params.layerSizes.back();
		hiddenSizes.assign(params.layerSizes.begin() + 1, params.layerSizes.end() - 1);
		hiddenActivations.assign(params.layerActivations.begin(), params.layerActivations.end() - 1);
		outputActivation = params.layerActivations.back();
		status = "Network loaded";
		return true;
	};

	auto LoadNetwork = [&] {
		Net loaded(netPath.s);
		if (loaded.GetNodeCount() == 0) {
			status = "Could not load network";
			return false;
		}
		net = loaded;
		return SyncArchitecture();
	};

	auto LoadProfile = [&] {
		if (!profile.Load(profilePath.s)) {
			status = "Could not read profile";
			return false;
		}
		const std::string dir = DirOf(profilePath.s);
		auto abs = [&](const std::string &p) { return Buf((IsAbs(p) ? p : dir + p).c_str()); };
		netPath = abs(profile.network);
		trainX = abs(profile.trainData);
		trainY = abs(profile.trainLabels);
		testX = abs(profile.testData);
		testY = abs(profile.testLabels);
		if (!profile.network.empty() && LoadNetwork()) {
			status = "Profile and network loaded";
			return true;
		}
		if (profile.network.empty()) status = "Profile loaded; no network specified";
		return profile.network.empty();
	};

	bool done = false;
	while (!done) {
		MSG msg;
		while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
			if (msg.message == WM_QUIT) done = true;
		}
		if (done) break;

		ImGui_ImplDX11_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();
		ImGui::Begin("Trainer");
		const bool running = trainer.IsRunning();

		if (ImGui::CollapsingHeader("Profile", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::InputText("Profile file", profilePath.s, sizeof(profilePath.s));
			if (ImGui::Button("Load profile")) {
				LoadProfile();
			}
			ImGui::SameLine();
			if (ImGui::Button("Save profile")) {
				profile.network = netPath.s; profile.trainData = trainX.s; profile.trainLabels = trainY.s;
				profile.testData = testX.s; profile.testLabels = testY.s;
				status = profile.Save(profilePath.s) ? "Profile saved" : "Could not write profile";
			}
			ImGui::InputText("Network", netPath.s, sizeof(netPath.s));
			ImGui::InputText("Train data", trainX.s, sizeof(trainX.s));
			ImGui::InputText("Train labels", trainY.s, sizeof(trainY.s));
			ImGui::InputText("Test data", testX.s, sizeof(testX.s));
			ImGui::InputText("Test labels", testY.s, sizeof(testY.s));
		}

		if (ImGui::CollapsingHeader("Network", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::BeginDisabled(running);
			ImGui::Text("Inputs: %d", inSize);
			static const char *activationNames[] = { "Linear", "Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "Sine" };
			for (size_t i = 0; i < hiddenSizes.size(); ++i) {
				ImGui::PushID((int)i);
				ImGui::Text("Hidden layer %d", (int)i + 1);
				ImGui::SameLine();
				ImGui::SetNextItemWidth(100.f);
				ImGui::InputInt("Width", &hiddenSizes[i]);
				ImGui::SameLine();
				ImGui::SetNextItemWidth(130.f);
				int activation = (int)hiddenActivations[i];
				if (ImGui::Combo("Activation", &activation, activationNames, IM_ARRAYSIZE(activationNames))) {
					hiddenActivations[i] = static_cast<Activation>(activation);
				}
				ImGui::SameLine();
				const bool removeLayer = ImGui::SmallButton("Remove");
				ImGui::PopID();
				if (removeLayer) {
					hiddenSizes.erase(hiddenSizes.begin() + i);
					hiddenActivations.erase(hiddenActivations.begin() + i);
					break;
				}
			}
			if (ImGui::Button("Add hidden layer")) {
				hiddenSizes.push_back(hiddenSizes.empty() ? 8 : hiddenSizes.back());
				hiddenActivations.push_back(Tanh);
			}
			ImGui::Text("Outputs: %d", outSize);
			ImGui::SetNextItemWidth(130.f);
			int outputAct = (int)outputActivation;
			if (ImGui::Combo("Output activation", &outputAct, activationNames, IM_ARRAYSIZE(activationNames))) {
				outputActivation = static_cast<Activation>(outputAct);
			}
			if (ImGui::Button("Create network")) {
				const bool validWidths = std::all_of(hiddenSizes.begin(), hiddenSizes.end(), [](int width) { return width > 0; });
				if (inSize > 0 && outSize > 0 && !hiddenSizes.empty() && validWidths) {
					std::vector<Activation> activations = hiddenActivations;
					activations.push_back(outputActivation);
					net = Net(inSize, hiddenSizes, outSize, activations);
					status = "New network created";
				} else {
					status = "Network needs positive dimensions and at least one hidden layer";
				}
			}
			ImGui::SameLine();
			if (ImGui::Button("Load network")) LoadNetwork();
			ImGui::EndDisabled();
			ImGui::SameLine();
			if (ImGui::Button("Save network")) {
				trainer.WithNetwork([&] { net.SaveNetwork(netPath.s); });
				status = "Network saved";
			}
		}

		if (ImGui::CollapsingHeader("Training", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::InputFloat("Learning rate", &profile.learningRate);
			ImGui::InputFloat("Reg term", &profile.regTerm);
			ImGui::InputFloat("Weight scale", &profile.weightScale);
			ImGui::BeginDisabled(running);
			ImGui::InputInt("Batch count", &profile.batchCount);
			profile.batchCount = (std::max)(1, profile.batchCount);
			static const char *shuffleNames[] = { "None (sequential)", "Random shuffle", "Sliding window" };
			int shuffleIndex = (int)profile.shuffle;
			if (ImGui::Combo("Batch order", &shuffleIndex, shuffleNames, IM_ARRAYSIZE(shuffleNames))) {
				profile.shuffle = static_cast<NetBatchShuffleType>(shuffleIndex);
			}
			ImGui::EndDisabled();
			ImGui::BeginDisabled(running);
			if (ImGui::RadioButton("CPU", backend == TrainerBackend::Cpu)) backend = TrainerBackend::Cpu;
			ImGui::SameLine();
			ImGui::BeginDisabled(!gpuAvailable);
			if (ImGui::RadioButton("GPU (CUDA)", backend == TrainerBackend::Gpu)) backend = TrainerBackend::Gpu;
			ImGui::EndDisabled();
			if (!gpuAvailable) {
				ImGui::SameLine();
				ImGui::TextUnformatted("(no CUDA device found)");
			}
			ImGui::EndDisabled();
			if (!running) {
				if (ImGui::Button("Start")) {
					Eigen::MatrixXf X = LoadMatrix(trainX.s), Y = LoadMatrix(trainY.s);
					Eigen::MatrixXf tX, tY;
					/*if (testX.s[0] && testY.s[0]) { tX = LoadMatrix(testX.s); tY = LoadMatrix(testY.s); }*/
					NetBatchParams batch;
					batch.batchCount = profile.batchCount;
					batch.shuffleType = profile.shuffle;
					const int samples = (int)X.cols();
					const bool started = trainer.Start(&net, X, Y, tX, tY, profile.weightScale, profile.learningRate,
						profile.regTerm, backend, batch);
					if (started) {
						status = std::string(backend == TrainerBackend::Gpu ? "Training (GPU)" : "Training (CPU)") +
							", " + std::to_string(batch.batchCount) + " batch(es) of " +
							std::to_string(samples / batch.batchCount) + " samples";
					} else {
						status = "Cannot start: need a network, matching train data/labels and 1 <= batch count <= samples";
					}
				}
			} else if (trainer.IsStopping()) {
				ImGui::BeginDisabled();
				ImGui::Button("Stopping...");
				ImGui::EndDisabled();
			} else if (ImGui::Button("Stop")) {
				trainer.Stop();
				status = "Stop requested; finishing current epoch";
			}
			const std::vector<float> tr = trainer.TrainHistory(), te = trainer.TestHistory();
			if (!tr.empty()) ImGui::PlotLines("Train cost", tr.data(), (int)tr.size(), 0, nullptr, 0.f, FLT_MAX, ImVec2(0, 300));
			if (!te.empty()) ImGui::PlotLines("Test cost", te.data(), (int)te.size(), 0, nullptr, 0.f, FLT_MAX, ImVec2(0, 300));
			if (!tr.empty()) {
				ImGui::Text("Epochs: %d  average: %.2f epochs/sec  train: %.5f  test: %.5f",
					(int)tr.size(), trainer.EpochsPerSecond(), tr.back(), te.empty() ? 0.f : te.back());
			}
		}
		ImGui::TextUnformatted(status.c_str());
		ImGui::End();

		ImGui::Render();
		const float clear[4] = { 0.1f, 0.1f, 0.12f, 1.f };
		g_ctx->OMSetRenderTargets(1, &g_rtv, nullptr);
		g_ctx->ClearRenderTargetView(g_rtv, clear);
		ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
		g_swap->Present(1, 0);
	}

	trainer.Stop();
	ImGui_ImplDX11_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
	ReleaseRTV();
	g_swap->Release(); g_ctx->Release(); g_dev->Release();
	DestroyWindow(hwnd);
	UnregisterClass(wc.lpszClassName, inst);
	return 0;
}
