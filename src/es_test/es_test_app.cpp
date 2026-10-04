#include "es_test_app.h"
using namespace std;
using namespace Eigen;
static vector<string> categories;
static vector<string> headers;
static vector<string> prefixes;
static vector<string> functionNames;
static vector<vector<string>> arguments;

void PrintFile(const string fName) {
	std::string line;
	ifstream filePrint(fName);
	while (filePrint.good()) {
		std::getline(filePrint, line);
		cout << line << endl;
	}
}

void ReadTestList(const string fName) {
	cout << "Reading file " << fName << endl;
	std::string line;
	ifstream file(fName);
	TestParseState state = TestParseState::none;
	vector<string> currentFuncArgs;
	while (file.good()) {
		getline(file, line);
		stringstream iss(line);
		string val, lastVal;
		while (iss.good()) {
			std::getline(iss, val, ':');
			if (state == TestParseState::none) {
				val.erase(remove(val.begin(), val.end(), ' '), val.end());
				if (strFind(lastVal, "category")) {
					state = TestParseState::category;
					categories.push_back(val);
				}
				else if (strFind(lastVal, "header")) {
					state = TestParseState::header;
					headers.push_back(val);
				}
				else if (strFind(lastVal, "prefix")) {
					state = TestParseState::prefix;
					prefixes.push_back(val);
				}
				else if (strFind(lastVal, "functionName")) {
					state = TestParseState::functionName;
					functionNames.push_back(val);
				}
				else if (strFind(lastVal, "arguments")) {
					state = TestParseState::args;
				}
				lastVal = val;
			}
			switch (state) {
			case TestParseState::none:
			case TestParseState::header:
			case TestParseState::prefix:
			case TestParseState::functionName: //fall through
				state = TestParseState::none;
				break;
			case TestParseState::args: {
				if (strFind(val, "]")) {
					state = TestParseState::none;
					arguments.push_back(currentFuncArgs);
					currentFuncArgs.clear();
				}
				else {
					if (!strFind(val, "[")) {
						currentFuncArgs.push_back(strRemoveSpaces(val));
					}
				}
			} break;
			default:
				state = TestParseState::none;
				break;
			}
		}
	}
	PrintFile(fName);
	cout << "Finished reading file " << fName << endl;
	file.close();
}
void CreateGeneratedUnit(const string fName) {
	ofstream file(fName.c_str());
	file.clear();
	file << "//GENERATED FILE" << endl;
	string lastCategory = "";
	for (int fn = 0; fn < functionNames.size(); fn++) {
		string category = categories[fn];
		if (lastCategory == "") {
			file << "namespace " << category << " {" << endl;
		}
		else if (category.compare(lastCategory) != 0) {
			file << "}" << endl;
			file << "namespace " << category << " {" << endl;
		}
		string className = headers[fn];
		file << "TEST_CLASS(" << className << ") { public:" << endl;
		for (int arg = 0; arg < arguments[fn].size(); arg++) {
			string spacer = arg <= 9 ? "0" : "";
			file << "\tNAME_RUN(" << prefixes[fn] << spacer << arg << "_";
			string str = strRemoveSpaces(arguments[fn][arg]);
			str = strReplace(str, ".", "p");
			str = strReplace(str, ",", "x");
			str = strRemove(str, { '{' ,'}' ,'(' ,')' });
			file << str << ",";
			file << functionNames[fn] << "(";
			file << arguments[fn][arg];
			file << "));" << endl;
		}
		file << "};" << endl;
		lastCategory = category;
	}
	file << "}" << endl;
	file.close();
	PrintFile(fName);
}
void CreateGeneratedCpp(const string fName) {
	ofstream file(fName.c_str());
	file.clear();
	file << "//GENERATED FILE" << endl;
	for (int fn = 0; fn < functionNames.size(); fn++) {
		file << "PrintHeader(\"" << headers[fn] << "\");" << endl;
		for (int arg = 0; arg < arguments[fn].size(); arg++) {
			file << functionNames[fn] << "(";
			file << arguments[fn][arg];
			file << ");" << endl;
		}
	}
	file.close();
	PrintFile(fName);
}

void CreateGeneratedGTest(const string fName) {
	ofstream file(fName.c_str());
	file << "//GENERATED FILE" << endl;
	for (int fn = 0; fn < functionNames.size(); fn++) {
		for (int arg = 0; arg < arguments[fn].size(); arg++) {
			string spacer = arg <= 9 ? "0" : "";
			string str = strRemoveSpaces(arguments[fn][arg]);
			str = strReplace(str, ".", "p");
			str = strReplace(str, ",", "x");
			str = strReplace(str, "-", "n");
			str = strRemove(str, { '{' ,'}' ,'(' ,')' });
			file << "GTEST_RUN(" << categories[fn] << "_" << headers[fn] << ", ";
			file << prefixes[fn] << spacer << arg << "_" << str << ", ";
			file << functionNames[fn] << "(" << arguments[fn][arg] << "));" << endl;
		}
	}
	file.close();
	PrintFile(fName);
}

void RunAllTests() {
	initParallel();
	setNbThreads(4);
	verbosity = 2;
#ifndef TEST_LISTS
#define TEST_LISTS
#include GENERATED_TESTS
#endif
}

int main(int argc, char** argv) {
	if (argc > 1) {
		for (int i = 0; i <= argc; ++i) {
			if (strcmp(argv[i], "-b") == 0) {
				if (i + 3 > argc) {
					cout << "-b flag takes 3 args (+1 optional): .list file, cpp.generated, unit.generated, [gtest.generated]" << endl;
					return 0;
				}
				cout << "Building File" << endl;
				ReadTestList(argv[i + 1]);
				CreateGeneratedCpp(argv[i + 2]);
				CreateGeneratedUnit(argv[i + 3]);
				if (i + 4 < argc) {
					CreateGeneratedGTest(argv[i + 4]);
				}
			}
			else if (strcmp(argv[i], "-a") == 0) {
				RunAllTests();
			}
		}
		return 0;
	}
	cout << "(T)est individual, Test (A)ll, (B)uild, or e(X)it: ";
	HANDLE handle = GetStdHandle(STD_INPUT_HANDLE);
	DWORD events;
	INPUT_RECORD buffer;
	while (1) {
		PeekConsoleInput(handle, &buffer, 1, &events);
		if (events > 0) {
			ReadConsoleInput(handle, &buffer, 1, &events);
			WORD in = buffer.Event.KeyEvent.wVirtualKeyCode;
			if (in == 66) { //'b'
				cout << "Building File" << endl;
				ReadTestList("tests.list");
				CreateGeneratedCpp(GENERATED_TESTS);
				CreateGeneratedUnit(GENERATED_UNIT_TESTS);
				CreateGeneratedGTest(GENERATED_GTEST_TESTS);
				break;
			}
			else if (in == 65) { //'a'
				cout << "Running All Tests" << endl;
				RunAllTests();
				break;
			}
			else if (in == 84) { //'t'
				cout << "Test " << endl;
				ifstream file(GENERATED_TESTS);
				string line;
				int lineNumber = 0;
				while (file.good()) {
					std::getline(file, line);
					if (line != "") {
						string spacer = lineNumber <= 9 ? "  " : " ";
						cout << spacer << lineNumber << ": " << line << endl;
						lineNumber++;
					}
				}
				string inTest;
				cin >> inTest;
				int temp;
				strCast(&temp, inTest);
				break;
			}
			else if (in == 88) { //'x'
				break;
			}
		}
	}
	return 0;
}