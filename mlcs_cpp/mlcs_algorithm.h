/**
 * Enhanced Linear Multiple Longest Common Subsequence (MLCS) algorithm implementation.
 * Optimized C++ version for both character-level and token-level comparison.
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <regex>
#include <map>
#include <sstream>

class MLCSAlgorithm {
private:
    std::string language;
    std::unordered_map<std::string, std::vector<std::string>> russianEndings;
    std::unordered_map<std::string, std::vector<std::string>> englishEndings;
    std::unordered_map<std::string, std::vector<std::string>> educationalMarkers;
    std::unordered_map<std::string, std::regex> educationalMarkersRegex;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> domainKeywords;

    void loadLanguageResources();
    bool isDomainTerm(const std::string& token);
    std::vector<std::string> generateRussianVariants(const std::string& text);
    std::vector<std::string> generateEnglishVariants(const std::string& text);
    std::vector<std::string> lcsEfficient(const std::vector<std::string>& seq1, const std::vector<std::string>& seq2);
    std::vector<std::string> lcs(const std::vector<std::string>& seq1, const std::vector<std::string>& seq2);
    std::vector<std::string> findMlcsCharacter(const std::vector<std::vector<std::string>>& sequences, int minLength);
    std::vector<std::string> findMlcsToken(const std::vector<std::vector<std::string>>& sequences, int minLength);
    std::vector<std::string> findMlcsLinear(const std::vector<std::vector<std::string>>& sequences, int minLength);
    std::vector<std::pair<std::vector<std::string>, float>> findSignificantPatterns(
        const std::vector<std::string>& texts,
        int minLength,
        int minFrequency,
        const std::string& language);

public:
    MLCSAlgorithm(const std::string& lang = "en");

    std::string normalizeToken(const std::string& token, const std::string& language);
    std::unordered_set<std::string> generateVariants(const std::string& text);
    float matchVariants(const std::string& text, const std::string& target);
    std::vector<std::string> findMlcs(const std::vector<std::vector<std::string>>& sequences, int minLength = 2);
    std::vector<std::string> preprocessText(const std::string& text, const std::string& language = "");
    std::pair<std::vector<std::string>, float> extractConceptSignature(
        const std::string& conceptText,
        const std::vector<std::string>& contexts,
        const std::string& language = ""
    );
};
