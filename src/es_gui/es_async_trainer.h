#pragma once
#include "../es_core/stdNetTrainer.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

// Runs training on a worker thread so the GUI thread never blocks.
// The GUI polls History()/status; network access from the GUI goes through
// WithNetwork() so it never races with an epoch in flight.
class AsyncTrainer {
public:
	~AsyncTrainer() { Stop(); }

	// Takes ownership of nothing; `net` must outlive the trainer.
	bool Start(Net *net, const Eigen::MatrixXf &trainX, const Eigen::MatrixXf &trainY,
	           const Eigen::MatrixXf &testX, const Eigen::MatrixXf &testY,
	           float weightScale, float learnRate, float regTerm) {
		Stop();
		if (!net || net->GetNodeCount() == 0 || trainX.size() == 0 || trainY.size() == 0) return false;
		{
			std::lock_guard<std::mutex> l(mutex);
			trainCost.clear();
			testCost.clear();
			network = net;
			testData = testX;
			testLabels = testY;
			trainer = NetTrainer(net, trainX, trainY, weightScale, learnRate, regTerm);
		}
		stopRequested = false;
		running = true;
		worker = std::thread([this] { Run(); });
		return true;
	}

	void Stop() {
		stopRequested = true;
		if (worker.joinable()) worker.join();
		running = false;
	}

	bool IsRunning() const { return running; }

	// Thread safe snapshots of the per-epoch cost, for plotting histograms.
	std::vector<float> TrainHistory() { std::lock_guard<std::mutex> l(mutex); return trainCost; }
	std::vector<float> TestHistory() { std::lock_guard<std::mutex> l(mutex); return testCost; }

	template <class F>
	void WithNetwork(F &&f) { std::lock_guard<std::mutex> l(mutex); f(); }

private:
	void Run() {
		while (!stopRequested) {
			std::lock_guard<std::mutex> l(mutex);
			trainer.TrainSingleEpoch();
			trainCost.push_back(trainer.GetCache().cost);
			if (testData.size() && testLabels.size()) {
				const Eigen::MatrixXf diff = network->ForwardPropagation(testData) - testLabels;
				testCost.push_back(diff.squaredNorm() / float(testLabels.cols()));
			}
		}
		running = false;
	}

	std::thread worker;
	std::mutex mutex;
	std::atomic<bool> running{ false };
	std::atomic<bool> stopRequested{ false };
	Net *network = nullptr;
	NetTrainer trainer;
	Eigen::MatrixXf testData, testLabels;
	std::vector<float> trainCost, testCost;
};
