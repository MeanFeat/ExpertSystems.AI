// Dear ImGui (Win32 + DirectX11) front end for training networks.
#include "es_core_pch.h"
#include "stdMatrix.h"
#include "es_async_trainer.h"
#include "es_profile.h"
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
	Buf profilePath("profile.esprofile"), netPath("net.json");
	Buf trainX, trainY, testX, testY;
	int inSize = 2, outSize = 1, hidden = 8, layers = 2;
	std::string status = "Ready";

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
				if (profile.Load(profilePath.s)) {
					const std::string dir = DirOf(profilePath.s);
					auto abs = [&](const std::string &p) { return Buf((IsAbs(p) ? p : dir + p).c_str()); };
					netPath = abs(profile.network); trainX = abs(profile.trainData); trainY = abs(profile.trainLabels);
					testX = abs(profile.testData); testY = abs(profile.testLabels);
					status = "Profile loaded";
				} else status = "Could not read profile";
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
			ImGui::InputInt("Inputs", &inSize);
			ImGui::InputInt("Outputs", &outSize);
			ImGui::InputInt("Hidden layers", &layers);
			ImGui::InputInt("Hidden size", &hidden);
			if (ImGui::Button("New network") && inSize > 0 && outSize > 0 && hidden > 0 && layers >= 0) {
				std::vector<int> h(layers, hidden);
				std::vector<Activation> a(layers + 1, Tanh);
				a.back() = Linear;
				net = Net(inSize, h, outSize, a);
				status = "New network created";
			}
			ImGui::SameLine();
			if (ImGui::Button("Load network")) { net.LoadNetwork(netPath.s); status = "Network loaded"; }
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
			if (!running) {
				if (ImGui::Button("Start")) {
					Eigen::MatrixXf X = Eigen::BuildMatFromFile(trainX.s), Y = Eigen::BuildMatFromFile(trainY.s);
					Eigen::MatrixXf tX, tY;
					if (testX.s[0] && testY.s[0]) { tX = Eigen::BuildMatFromFile(testX.s); tY = Eigen::BuildMatFromFile(testY.s); }
					status = trainer.Start(&net, X, Y, tX, tY, profile.weightScale, profile.learningRate, profile.regTerm)
						? "Training" : "Cannot start: need a network and train data/labels";
				}
			} else if (ImGui::Button("Stop")) {
				trainer.Stop();
				status = "Stopped";
			}
			const std::vector<float> tr = trainer.TrainHistory(), te = trainer.TestHistory();
			if (!tr.empty()) ImGui::PlotHistogram("Train cost", tr.data(), (int)tr.size(), 0, nullptr, 0.f, FLT_MAX, ImVec2(0, 80));
			if (!te.empty()) ImGui::PlotHistogram("Test cost", te.data(), (int)te.size(), 0, nullptr, 0.f, FLT_MAX, ImVec2(0, 80));
			if (!tr.empty()) ImGui::Text("Epochs: %d  train: %.5f  test: %.5f", (int)tr.size(), tr.back(), te.empty() ? 0.f : te.back());
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
