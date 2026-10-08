#pragma once
#include "../es_core/stdNetTrainer.h"
#include "../es_core/d_NetTrainer.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

enum class TrainerBackend { Cpu, Gpu };

inline bool IsGpuAvailable() {
	int count = 0;
	return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Runs training on a worker thread so the GUI thread never blocks.
// The GUI polls History()/status; network access from the GUI goes through
// WithNetwork() so it never races with an epoch in flight.
class AsyncTrainer {
public:
	~AsyncTrainer() {
		Stop();
		JoinWorker();
	}

	// Takes ownership of nothing; `net` must outlive the trainer.
	bool Start(Net *net, const Eigen::MatrixXf &trainX, const Eigen::MatrixXf &trainY,
	           const Eigen::MatrixXf &testX, const Eigen::MatrixXf &testY,
	           float weightScale, float learnRate, float regTerm, const TrainerBackend trainerBackend = TrainerBackend::Cpu,
	           const NetBatchParams &batchParameters = NetBatchParams()) {
		Stop();
		JoinWorker();
		if (!net || net->GetNodeCount() == 0 || trainX.size() == 0 || trainY.size() == 0) return false;
		if (trainX.cols() != trainY.cols()) return false;
		if (batchParameters.batchCount < 1 || batchParameters.batchCount > trainX.cols()) return false;
		if (trainerBackend == TrainerBackend::Gpu && !IsGpuAvailable()) return false;
		{
			std::lock_guard<std::mutex> l(mutex);
			network = net;
			testData = testX;
			testLabels = testY;
			backend = trainerBackend;
			gpuTrainer.reset();
			if (backend == TrainerBackend::Gpu) {
				gpuTrainer = std::make_unique<d_NetTrainer>(net, trainX, trainY, weightScale, learnRate, regTerm,
					d_NetBatchParams(batchParameters.batchCount, batchParameters.shuffleType));
			} else {
				cpuTrainer = NetTrainer(net, trainX, trainY, weightScale, learnRate, regTerm, batchParameters);
			}
		}
		{
			std::lock_guard<std::mutex> l(historyMutex);
			trainCost.clear();
			testCost.clear();
		}
		stopRequested = false;
		running = true;
		startTime = std::chrono::steady_clock::now();
		epochsPerSecond = 0.f;
		worker = std::thread([this] { Run(); });
		return true;
	}

	void Stop() {
		stopRequested = true;
	}

	bool IsRunning() const { return running; }
	bool IsStopping() const { return running && stopRequested; }
	float EpochsPerSecond() const { return epochsPerSecond; }

	// Thread safe snapshots of the per-epoch cost, for plotting histograms.
	std::vector<float> TrainHistory() { std::lock_guard<std::mutex> l(historyMutex); return trainCost; }
	std::vector<float> TestHistory() { std::lock_guard<std::mutex> l(historyMutex); return testCost; }

	template <class F>
	void WithNetwork(F &&f) {
		std::lock_guard<std::mutex> l(mutex);
		SyncHostNetwork();
		f();
	}

private:
	// GPU training keeps the weights on the device; copy them back before the host network is read.
	void SyncHostNetwork() {
		if (backend == TrainerBackend::Gpu && gpuTrainer) gpuTrainer->RefreshHostNetwork();
	}

	void TrainEpoch() {
		if (backend == TrainerBackend::Gpu) {
			gpuTrainer->TrainSingleEpoch();
			lastTrainCost = gpuTrainer->GetCache().cost;
		} else {
			cpuTrainer.TrainSingleEpoch();
			lastTrainCost = cpuTrainer.GetCache().cost;
		}
	}

	void JoinWorker() {
		if (worker.joinable()) worker.join();
	}

	void Run() {
		size_t completedEpochs = 0;
		while (!stopRequested) {
			float trainCostForEpoch;
			float testCostForEpoch = 0.f;
			bool hasTestData = false;
			{
				std::lock_guard<std::mutex> l(mutex);
				TrainEpoch();
				trainCostForEpoch = lastTrainCost;
				if (testData.size() && testLabels.size()) {
					SyncHostNetwork();
					const Eigen::MatrixXf diff = network->ForwardPropagation(testData) - testLabels;
					testCostForEpoch = diff.squaredNorm() / float(testLabels.cols());
					hasTestData = true;
				}
			}
			{
				std::lock_guard<std::mutex> l(historyMutex);
				trainCost.push_back(trainCostForEpoch);
				if (hasTestData) testCost.push_back(testCostForEpoch);
			}
			++completedEpochs;
			const std::chrono::duration<float> elapsed = std::chrono::steady_clock::now() - startTime;
			if (elapsed.count() > 0.f) epochsPerSecond = float(completedEpochs) / elapsed.count();
		}
		{
			std::lock_guard<std::mutex> l(mutex);
			SyncHostNetwork();
		}
		running = false;
	}

	std::thread worker;
	std::mutex mutex;
	std::mutex historyMutex;
	std::atomic<bool> running{ false };
	std::atomic<bool> stopRequested{ false };
	std::atomic<float> epochsPerSecond{ 0.f };
	std::chrono::steady_clock::time_point startTime;
	Net *network = nullptr;
	TrainerBackend backend = TrainerBackend::Cpu;
	NetTrainer cpuTrainer;
	std::unique_ptr<d_NetTrainer> gpuTrainer;
	float lastTrainCost = 0.f;
	Eigen::MatrixXf testData, testLabels;
	std::vector<float> trainCost, testCost;
};
