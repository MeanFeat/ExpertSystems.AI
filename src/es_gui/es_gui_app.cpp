// Dear ImGui (Win32 + DirectX11) front end for training networks.
#include "es_core_pch.h"
#include "stdMatrix.h"
#include "es_async_trainer.h"
#include "es_profile.h"
#include <algorithm>
#include <d3d11.h>
#include <commdlg.h>
#pragma comment(lib, "comdlg32.lib")
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

static std::string FullPath(const std::string &p) {
	char out[MAX_PATH];
	const DWORD n = GetFullPathNameA(p.empty() ? "." : p.c_str(), MAX_PATH, out, nullptr);
	std::string r = (n > 0 && n < MAX_PATH) ? std::string(out) : p;
	for (char &c : r) if (c == '/') c = '\\';
	return r;
}

// Data and network files are stored relative to the profile's folder.
static std::string ResolveInProfile(const Buf &profilePath, const char *rel) {
	return rel[0] ? DirOf(profilePath.s) + rel : std::string();
}

// Converts a picked absolute path into a path relative to the profile folder.
// Fails if the file is not in the profile folder or one of its sub folders.
static bool MakeRelativeToProfile(const Buf &profilePath, const std::string &picked, Buf &rel) {
	std::string base = FullPath(DirOf(profilePath.s));
	if (base.empty() || base.back() != '\\') base += '\\';
	const std::string full = FullPath(picked);
	if (full.size() <= base.size() || _strnicmp(full.c_str(), base.c_str(), base.size()) != 0) return false;
	rel = Buf(full.c_str() + base.size());
	return true;
}

static Eigen::MatrixXf LoadMatrix(const char *path) {
	const std::string filePath(path);
	Eigen::MatrixXf matrix;
	if (filePath.size() >= 4 && _stricmp(filePath.c_str() + filePath.size() - 4, ".dat") == 0) {
		Eigen::read_binary(path, matrix);
	} else {
		matrix = Eigen::BuildMatFromFile(filePath);
	}
	return matrix;
}

// Opens the standard Windows Open/Save explorer dialog. Returns true if a path was chosen.
static bool BrowseFile(HWND owner, Buf &path, bool save, const char *filter) {
	char file[260];
	strncpy_s(file, path.s, _TRUNCATE);
	OPENFILENAMEA ofn = {};
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = owner;
	ofn.lpstrFilter = filter;
	ofn.lpstrFile = file;
	ofn.nMaxFile = sizeof(file);
	ofn.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST | (save ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);
	if (!(save ? GetSaveFileNameA(&ofn) : GetOpenFileNameA(&ofn))) return false;
	path = Buf(file);
	return true;
}

static const char *kProfileFilter = "Profiles (*.esprofile)\0*.esprofile\0All files\0*.*\0\0";
static const char *kNetFilter = "Networks (*.json)\0*.json\0All files\0*.*\0\0";
static const char *kDataFilter = "Data files (*.csv;*.dat)\0*.csv;*.dat\0CSV (*.csv)\0*.csv\0Binary (*.dat)\0*.dat\0All files\0*.*\0\0";

// Browses for a file that must live in the profile folder (or a sub folder); stores the relative path.
static bool BrowseRelative(HWND owner, const Buf &profilePath, Buf &rel, bool save, const char *filter, std::string &status) {
	Buf picked(ResolveInProfile(profilePath, rel.s).c_str());
	if (!BrowseFile(owner, picked, save, filter)) return false;
	if (!MakeRelativeToProfile(profilePath, picked.s, rel)) {
		status = "File must be in the profile folder or a sub folder";
		return false;
	}
	return true;
}

// Text field plus a "..." button that opens the explorer dialog.
static bool PathField(HWND owner, const Buf &profilePath, const char *label, Buf &path, bool save, const char *filter, std::string &status) {
	ImGui::PushID(label);
	ImGui::InputText(label, path.s, sizeof(path.s));
	ImGui::SameLine();
	const bool picked = ImGui::Button("...") && BrowseRelative(owner, profilePath, path, save, filter, status);
	ImGui::PopID();
	return picked;
}

int WINAPI WinMain(HINSTANCE inst, HINSTANCE, LPSTR, int) {
	WNDCLASSEX wc = { sizeof(wc), CS_CLASSDC, WndProc, 0, 0, inst, nullptr, nullptr, nullptr, nullptr, "esgui", nullptr };
	RegisterClassEx(&wc);
	HWND hwnd = CreateWindow(wc.lpszClassName, "ExpertSystems.AI Trainer", WS_OVERLAPPEDWINDOW, 100, 100, 1100, 1500, nullptr, nullptr, inst, nullptr);

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
		Net loaded(ResolveInProfile(profilePath, netPath.s));
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
		auto abs = [&](const std::string &p) { return Buf(p.c_str()); };
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
			if (ImGui::Button("Load profile") && BrowseFile(hwnd, profilePath, false, kProfileFilter)) {
				LoadProfile();
			}
			ImGui::SameLine();
			if (ImGui::Button("Save profile") && BrowseFile(hwnd, profilePath, true, kProfileFilter)) {
				profile.network = netPath.s; profile.trainData = trainX.s; profile.trainLabels = trainY.s;
				profile.testData = testX.s; profile.testLabels = testY.s;
				status = profile.Save(profilePath.s) ? "Profile saved" : "Could not write profile";
			}
			PathField(hwnd, profilePath, "Network", netPath, false, kNetFilter, status);
			PathField(hwnd, profilePath, "Train data", trainX, false, kDataFilter, status);
			PathField(hwnd, profilePath, "Train labels", trainY, false, kDataFilter, status);
			PathField(hwnd, profilePath, "Test data", testX, false, kDataFilter, status);
			PathField(hwnd, profilePath, "Test labels", testY, false, kDataFilter, status);
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
			if (ImGui::Button("Load network") && BrowseRelative(hwnd, profilePath, netPath, false, kNetFilter, status)) LoadNetwork();
			ImGui::EndDisabled();
			ImGui::SameLine();
			if (ImGui::Button("Save network") && BrowseRelative(hwnd, profilePath, netPath, true, kNetFilter, status)) {
				trainer.WithNetwork([&] { net.SaveNetwork(ResolveInProfile(profilePath, netPath.s)); });
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
					Eigen::MatrixXf X = LoadMatrix(ResolveInProfile(profilePath, trainX.s).c_str()), Y = LoadMatrix(ResolveInProfile(profilePath, trainY.s).c_str());
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
