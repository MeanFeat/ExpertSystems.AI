#pragma once
#include "es_core_pch.h"
#include <regex>

/**
 * String parsing and manipulation utilities
 */

#define OUT_LINE(f, t, s) file << strTab(t) << s << std::endl;

/**
 * Convert string to specified type
 * @param out Output pointer to store converted value
 * @param str Input string to convert
 */
template <class T>
void strCast(T *out, const std::string &str) {
	std::stringstream convertor(str);
	convertor >> *out;
}

/**
 * Check if string contains a token
 * @param str String to search in
 * @param token Token to search for
 * @return true if token is found
 */
inline bool strFind(const std::string &str, const std::string &token) {
	return str.find(token) != std::string::npos;
}

/**
 * Generate indentation string with tabs
 * @param num Number of tabs
 * @return String containing num tabs
 */
inline std::string strTab(const int num) {
	std::string outStr;
	outStr.reserve(num);  // Pre-allocate for efficiency
	for (int i = 0; i < num; i++) {
		outStr += "\t";
	}
	return outStr;
}

/**
 * Remove spaces and tabs from string
 * @param s Input string
 * @return String with spaces and tabs removed
 */
inline std::string strRemoveSpaces(std::string s){
	s.erase(remove(s.begin(), s.end(), ' '), s.end());
	s.erase(remove(s.begin(), s.end(), '\t'), s.end());
	return s;
}

/**
 * Remove all occurrences of a character from string
 * @param s Input string
 * @param r Character to remove
 * @return String with character removed
 */
inline std::string strRemove(std::string s, const char r){
	s.erase(remove(s.begin(), s.end(), r), s.end());
	return s;
}

/**
 * Remove all occurrences of multiple characters from string
 * @param s Input string
 * @param rs Vector of characters to remove
 * @return String with characters removed
 */
inline std::string strRemove(std::string s, const std::vector<char> &rs) {
	for (size_t i = 0; i < rs.size(); i++) {
		s = strRemove(s, rs[i]);
	}
	return s;
}

/**
 * Replace all occurrences of a substring
 * @param s Input string
 * @param f Substring to find
 * @param r Replacement string
 * @return String with replacements made
 */
inline std::string strReplace(std::string s, const std::string &f, const std::string &r){
	std::string out = s;
	size_t index = 0;
	const size_t findLength = f.length();
	while (true) {
		index = out.find(f, index);
		if (index == std::string::npos) break;
		out.replace(index, findLength, r);
		index += findLength;
	}
	return out;
}
