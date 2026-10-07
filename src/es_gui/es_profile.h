#pragma once
#include <fstream>
#include <sstream>
#include <string>

// A profile bundles everything needed for a training session into one small
// text file (*.esprofile) of `key=value` lines. Paths are stored relative to
// the profile so a profile folder can be moved, or exported from / imported
// into Unreal Engine (e.g. by the ProcAnim plugin) without edits.
//
//   network=net.json          network file (Net::SaveNetwork format)
//   train_data=train_x.csv    comma separated, see Eigen::BuildMatFromFile
//   train_labels=train_y.csv
//   test_data=test_x.csv
//   test_labels=test_y.csv
//   learning_rate=2
//   reg_term=20
//   weight_scale=1
struct EsProfile {
	std::string network, trainData, trainLabels, testData, testLabels;
	float learningRate = 2.f;
	float regTerm = 20.f;
	float weightScale = 1.f;

	static std::string Trim(const std::string &s) {
		const size_t a = s.find_first_not_of(" \t\r\n");
		if (a == std::string::npos) return "";
		return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
	}

	bool Save(const std::string &path) const {
		std::ofstream f(path);
		if (!f.good()) return false;
		f << "network=" << network << "\n"
		  << "train_data=" << trainData << "\n"
		  << "train_labels=" << trainLabels << "\n"
		  << "test_data=" << testData << "\n"
		  << "test_labels=" << testLabels << "\n"
		  << "learning_rate=" << learningRate << "\n"
		  << "reg_term=" << regTerm << "\n"
		  << "weight_scale=" << weightScale << "\n";
		return f.good();
	}

	bool Load(const std::string &path) {
		std::ifstream f(path);
		if (!f.good()) return false;
		std::string line;
		while (std::getline(f, line)) {
			const size_t eq = line.find('=');
			if (eq == std::string::npos) continue;
			const std::string k = Trim(line.substr(0, eq)), v = Trim(line.substr(eq + 1));
			if (k == "network") network = v;
			else if (k == "train_data") trainData = v;
			else if (k == "train_labels") trainLabels = v;
			else if (k == "test_data") testData = v;
			else if (k == "test_labels") testLabels = v;
			else if (k == "learning_rate") learningRate = (float)atof(v.c_str());
			else if (k == "reg_term") regTerm = (float)atof(v.c_str());
			else if (k == "weight_scale") weightScale = (float)atof(v.c_str());
		}
		return true;
	}
};
