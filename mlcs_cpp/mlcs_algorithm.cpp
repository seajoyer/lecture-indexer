/**
 * Enhanced Linear Multiple Longest Common Subsequence (MLCS) algorithm implementation.
 * Optimized C++ version for both character-level and token-level comparison.
 */

#include "mlcs_algorithm.h"
#include <algorithm>
#include <cctype>
#include <set>
#include <map>
#include <iostream>
#include <sstream>

// Constructor
MLCSAlgorithm::MLCSAlgorithm(const std::string& lang) : language(lang) {
    // Initialize language resources
    loadLanguageResources();
}

// Load language resources for preprocessing
void MLCSAlgorithm::loadLanguageResources() {
    // Russian word endings for morphological variants
    russianEndings = {
        {"ие", {"ия", "ий", "ием"}},
        {"ия", {"ие", "ий", "ию", "ией"}},
        {"ть", {"ти"}},
        {"ость", {"ости", "остей", "остью"}},
        {"ство", {"ства", "ствам"}},
        {"ние", {"ния", "ний", "нием"}},
        {"а", {"ы", "у", "е"}},
        {"я", {"и", "ю", "е"}},
        {"й", {"я", "ю", "и", "ем"}},
        {"ь", {"и", "ей", "ью"}}
    };

    // English plurals and verb forms
    englishEndings = {
        {"s", {""}},  // plural -> singular
        {"", {"s"}},  // singular -> plural
        {"ing", {"", "e"}},  // running -> run, rune
        {"ed", {"", "e"}}    // played -> play, playe
    };

    // Educational content markers
    educationalMarkers = {
        {"en", {
            "important concept",
            "key principle",
            "fundamental idea",
            "essential to understand",
            "core concept",
            "critical to",
            "central idea",
            "primarily concerned with",
            "focuses on",
            "the main",
            "in depth",
            "thoroughly",
            "explain in detail",
            "explore the",
            "analyze",
            "examine",
            "investigate",
            "detailed",
            "significant",
            "important"
        }},
        {"ru", {
            "важная концепция",
            "ключевой принцип",
            "фундаментальная идея",
            "необходимо понять",
            "основная концепция",
            "критически важно",
            "центральная идея",
            "в первую очередь",
            "фокусируется на",
            "главный",
            "подробно",
            "тщательно",
            "объяснить детально",
            "исследовать",
            "анализировать",
            "изучить",
            "исследовать",
            "детальный",
            "значительный",
            "важный"
        }}
    };

    // Initialize educational markers regex
    for (const auto& [lang, patterns] : educationalMarkers) {
        std::string combined;
        for (size_t i = 0; i < patterns.size(); ++i) {
            combined += patterns[i];
            if (i < patterns.size() - 1) {
                combined += "|";
            }
        }
        educationalMarkersRegex[lang] = std::regex(combined, std::regex_constants::icase);
    }

    // Domain-specific keywords
    domainKeywords = {
        {"physics", {
            {"en", {
                "quantum", "mechanics", "wave", "function", "operator", "state",
                "eigenvalue", "eigenstate", "hamiltonian", "commutator", "hermitian",
                "observable", "measurement", "probability", "amplitude", "schrodinger",
                "dirac", "bra", "ket", "hilbert", "space", "vector", "momentum", "energy",
                "position", "uncertainty", "principle", "entanglement", "superposition",
                "fermion", "boson", "photon", "electron", "proton", "neutron",
                "spin", "charge", "field", "potential", "barrier", "well",
                "particle", "duality", "interference", "diffraction"
            }},
            {"ru", {
                "квантовый", "квантовая", "квантовое", "квантовые", "квантовость",
                "механика", "волновая", "функция", "оператор", "состояние",
                "собственное", "значение", "собственный", "вектор", "собственная",
                "гамильтониан", "коммутатор", "эрмитов", "эрмитово", "эрмитова",
                "наблюдаемая", "измерение", "вероятность", "амплитуда", "шредингер",
                "дирак", "бра", "кет", "гильбертово", "пространство",
                "импульс", "энергия", "положение", "координата", "координаты",
                "неопределенность", "принцип", "запутанность", "суперпозиция"
            }}
        }},
        {"mathematics", {
            {"en", {
                "function", "derivative", "integral", "differential", "equation",
                "theorem", "lemma", "proof", "corollary", "proposition", "axiom",
                "definition", "variable", "constant", "expression", "formula",
                "identity", "inequality", "transformation", "mapping", "morphism",
                "isomorphism", "homomorphism", "bijection", "surjection", "injection",
                "domain", "codomain", "range", "image", "kernel", "vector", "scalar",
                "tensor", "matrix", "determinant", "trace", "eigenvalue", "eigenvector"
            }},
            {"ru", {
                "функция", "производная", "интеграл", "дифференциал", "уравнение",
                "теорема", "лемма", "доказательство", "следствие", "предложение", "аксиома",
                "определение", "переменная", "постоянная", "выражение", "формула",
                "тождество", "неравенство", "преобразование", "отображение", "морфизм",
                "изоморфизм", "гомоморфизм", "биекция", "сюръекция", "инъекция",
                "область определения", "область значений", "образ", "ядро", "вектор", "скаляр",
                "тензор", "матрица", "определитель", "след", "собственное значение"
            }}
        }}
    };
}

bool MLCSAlgorithm::isDomainTerm(const std::string& token) {
    // Check if a token is a domain-specific term
    for (const auto& [domain, langKeywords] : domainKeywords) {
        // Check in current language
        auto langIt = langKeywords.find(language);
        if (langIt != langKeywords.end()) {
            if (std::find(langIt->second.begin(), langIt->second.end(), token) != langIt->second.end()) {
                return true;
            }
        }

        // Try English as fallback
        if (language != "en") {
            auto enIt = langKeywords.find("en");
            if (enIt != langKeywords.end()) {
                if (std::find(enIt->second.begin(), enIt->second.end(), token) != enIt->second.end()) {
                    return true;
                }
            }
        }
    }
    return false;
}

std::string MLCSAlgorithm::normalizeToken(const std::string& token, const std::string& language) {
    if (token.empty()) {
        return "";
    }

    // Use provided language or default
    std::string lang = language.empty() ? this->language : language;

    // Lowercase the token
    std::string normalized = token;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
        [](unsigned char c){ return std::tolower(c); });

    // Apply language-specific normalizations
    if (lang == "ru") {
        // Replace 'ё' with 'е' (common in Russian text normalization)
        size_t pos = 0;
        while ((pos = normalized.find(u8"ё", pos)) != std::string::npos) {
            normalized.replace(pos, 1, "е");
            pos += 1;
        }

        // Remove Russian soft sign (ь) and hard sign (ъ) at the end of words
        if (!normalized.empty() && (normalized.back() == 'ь' || normalized.back() == 'ъ')) {
            normalized.pop_back();
        }

        // Basic Russian normalization for endings (simplified)
        if (normalized.size() > 4) {
            if (normalized.substr(normalized.size() - 3) == "ого" ||
                normalized.substr(normalized.size() - 3) == "его") {
                normalized.replace(normalized.size() - 3, 3, "ый");
            }
            else if (normalized.substr(normalized.size() - 3) == "ому" ||
                     normalized.substr(normalized.size() - 3) == "ему") {
                normalized.replace(normalized.size() - 3, 3, "ый");
            }
            else if (normalized.substr(normalized.size() - 2) == "ую" ||
                     normalized.substr(normalized.size() - 2) == "юю") {
                normalized.replace(normalized.size() - 2, 2, "ая");
            }
        }
    }
    else {  // English and other languages
        // Remove common English suffixes for normalization
        if (normalized.size() > 3) {
            // Plurals and verb forms (basic stemming)
            if (normalized.size() > 2 && normalized.substr(normalized.size() - 1) == "s" &&
                !(normalized.size() > 3 && normalized.substr(normalized.size() - 2) == "ss")) {
                normalized.pop_back();
            }
            else if (normalized.size() > 3 && normalized.substr(normalized.size() - 2) == "es") {
                normalized.resize(normalized.size() - 2);
            }
            else if (normalized.size() > 4 && normalized.substr(normalized.size() - 3) == "ing") {
                normalized.resize(normalized.size() - 3);
            }
            else if (normalized.size() > 4 && normalized.substr(normalized.size() - 2) == "ed") {
                normalized.resize(normalized.size() - 2);
            }
        }
    }

    // Remove non-alphanumeric characters except for hyphens
    normalized.erase(
        std::remove_if(normalized.begin(), normalized.end(),
            [](char c) { return !std::isalnum(c) && c != '-'; }),
        normalized.end());

    return normalized;
}

std::vector<std::string> MLCSAlgorithm::preprocessText(const std::string& text, const std::string& language) {
    if (text.empty()) {
        return {};
    }

    // Use provided language or default
    std::string lang = language.empty() ? this->language : language;

    // Get all domain-specific keywords for better filtering decisions
    std::unordered_set<std::string> domainKeywordsSet;
    for (const auto& [domain, langKeywords] : domainKeywords) {
        auto langIt = langKeywords.find(lang);
        if (langIt != langKeywords.end()) {
            domainKeywordsSet.insert(langIt->second.begin(), langIt->second.end());
        }

        // Fallback to English keywords if language not available
        if (langIt == langKeywords.end()) {
            auto enIt = langKeywords.find("en");
            if (enIt != langKeywords.end()) {
                domainKeywordsSet.insert(enIt->second.begin(), enIt->second.end());
            }
        }
    }

    // Simple tokenization
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream stream(text);

    while (stream >> token) {
        std::transform(token.begin(), token.end(), token.begin(),
            [](unsigned char c){ return std::tolower(c); });

        std::string normalized = normalizeToken(token, lang);

        // Skip empty tokens
        if (normalized.empty()) {
            continue;
        }

        // Always keep domain keywords
        if (domainKeywordsSet.find(normalized) != domainKeywordsSet.end()) {
            tokens.push_back(normalized);
            continue;
        }

        // Basic stopword filtering
        static const std::unordered_set<std::string> enStopwords = {
            "the", "a", "an", "and", "or", "but", "if", "because", "as", "what",
            "which", "this", "that", "these", "those", "then", "just", "so", "than",
            "such", "both", "through", "about", "for", "is", "of", "while", "during"
        };

        static const std::unordered_set<std::string> ruStopwords = {
            "и", "в", "на", "с", "по", "к", "у", "от", "из", "для", "это", "так", "что", "как"
        };

        const auto& stopwords = (lang == "ru") ? ruStopwords : enStopwords;

        // Skip stopwords and very short tokens
        if (stopwords.find(normalized) != stopwords.end() || normalized.size() <= 2) {
            continue;
        }

        // Skip tokens that are just numbers
        bool isDigit = true;
        for (char c : normalized) {
            if (!std::isdigit(c)) {
                isDigit = false;
                break;
            }
        }

        if (isDigit) {
            continue;
        }

        tokens.push_back(normalized);
    }

    return tokens;
}

std::unordered_set<std::string> MLCSAlgorithm::generateVariants(const std::string& text) {
    std::unordered_set<std::string> variants;
    variants.insert(text);  // Always include the original

    std::vector<std::string> additionalVariants;
    if (language == "ru") {
        additionalVariants = generateRussianVariants(text);
    }
    else {  // Default to English
        additionalVariants = generateEnglishVariants(text);
    }

    for (const auto& variant : additionalVariants) {
        variants.insert(variant);
    }

    return variants;
}

std::vector<std::string> MLCSAlgorithm::generateRussianVariants(const std::string& text) {
    std::vector<std::string> variants;
    std::istringstream iss(text);
    std::vector<std::string> words;
    std::string word;

    // Split into words
    while (iss >> word) {
        words.push_back(word);
    }

    // For single words, apply ending transformations
    if (words.size() == 1) {
        word = words[0];
        for (const auto& [ending, replacements] : russianEndings) {
            if (word.size() > ending.size() + 2 &&
                word.substr(word.size() - ending.size()) == ending) {

                std::string base = word.substr(0, word.size() - ending.size());
                for (const auto& replacement : replacements) {
                    variants.push_back(base + replacement);
                }
            }
        }
    }
    // For multi-word terms like "соотношение неопределенности"
    else if (words.size() > 1) {
        // Often only the last word changes in Russian phrases
        std::string lastWord = words.back();
        for (const auto& [ending, replacements] : russianEndings) {
            if (lastWord.size() > ending.size() + 2 &&
                lastWord.substr(lastWord.size() - ending.size()) == ending) {

                std::string base = lastWord.substr(0, lastWord.size() - ending.size());
                for (const auto& replacement : replacements) {
                    std::vector<std::string> newWords = words;
                    newWords.back() = base + replacement;

                    std::string variant;
                    for (const auto& w : newWords) {
                        if (!variant.empty()) variant += " ";
                        variant += w;
                    }
                    variants.push_back(variant);
                }
            }
        }

        // Generate variants where the first word changes too
        std::string firstWord = words.front();
        for (const auto& [ending, replacements] : russianEndings) {
            if (firstWord.size() > ending.size() + 2 &&
                firstWord.substr(firstWord.size() - ending.size()) == ending) {

                std::string base = firstWord.substr(0, firstWord.size() - ending.size());
                for (const auto& replacement : replacements) {
                    std::vector<std::string> newWords = words;
                    newWords.front() = base + replacement;

                    std::string variant;
                    for (const auto& w : newWords) {
                        if (!variant.empty()) variant += " ";
                        variant += w;
                    }
                    variants.push_back(variant);
                }
            }
        }
    }

    return variants;
}

std::vector<std::string> MLCSAlgorithm::generateEnglishVariants(const std::string& text) {
    std::vector<std::string> variants;
    std::istringstream iss(text);
    std::vector<std::string> words;
    std::string word;

    // Split into words
    while (iss >> word) {
        words.push_back(word);
    }

    // For single words
    if (words.size() == 1) {
        word = words[0];
        for (const auto& [ending, replacements] : englishEndings) {
            if (word.size() > ending.size() + 2 &&
                word.substr(word.size() - ending.size()) == ending) {

                std::string base = word.substr(0, word.size() - ending.size());
                for (const auto& replacement : replacements) {
                    variants.push_back(base + replacement);
                }
            }
        }
    }
    // For multi-word phrases, try changing one word at a time
    else if (words.size() > 1) {
        // Try variants of the last word
        std::string lastWord = words.back();
        for (const auto& [ending, replacements] : englishEndings) {
            if (lastWord.size() > ending.size() + 2 &&
                lastWord.substr(lastWord.size() - ending.size()) == ending) {

                std::string base = lastWord.substr(0, lastWord.size() - ending.size());
                for (const auto& replacement : replacements) {
                    std::vector<std::string> newWords = words;
                    newWords.back() = base + replacement;

                    std::string variant;
                    for (const auto& w : newWords) {
                        if (!variant.empty()) variant += " ";
                        variant += w;
                    }
                    variants.push_back(variant);
                }
            }
        }
    }

    return variants;
}

float MLCSAlgorithm::matchVariants(const std::string& text, const std::string& target) {
    // If exact match, return perfect score
    if (text == target) {
        return 1.0;
    }

    // Make lowercase versions for better matching
    std::string textLower = text;
    std::string targetLower = target;
    std::transform(textLower.begin(), textLower.end(), textLower.begin(),
        [](unsigned char c){ return std::tolower(c); });
    std::transform(targetLower.begin(), targetLower.end(), targetLower.begin(),
        [](unsigned char c){ return std::tolower(c); });

    // If lowercase match, still consider it perfect
    if (textLower == targetLower) {
        return 1.0;
    }

    // Generate variants of both texts
    auto textVariants = generateVariants(textLower);
    auto targetVariants = generateVariants(targetLower);

    // Check for matches
    for (const auto& textVar : textVariants) {
        for (const auto& targetVar : targetVariants) {
            if (textVar == targetVar) {
                return 0.95;  // Very high but not perfect score for variant matches
            }
        }
    }

    // If words have common stems but different endings, return medium score
    std::istringstream issText(textLower);
    std::vector<std::string> textWords;
    std::string word;
    while (issText >> word) {
        textWords.push_back(word);
    }

    std::istringstream issTarget(targetLower);
    std::vector<std::string> targetWords;
    while (issTarget >> word) {
        targetWords.push_back(word);
    }

    if (textWords.size() == targetWords.size()) {
        // Check if all words except the last one match
        if (textWords.size() > 1) {
            bool prefixMatch = true;
            for (size_t i = 0; i < textWords.size() - 1; ++i) {
                if (textWords[i] != targetWords[i]) {
                    prefixMatch = false;
                    break;
                }
            }

            if (prefixMatch) {
                // Check if last words are morphological variants
                std::string lastText = textWords.back();
                std::string lastTarget = targetWords.back();

                // Check if they share a common stem (3+ characters)
                size_t minLen = std::min(lastText.size(), lastTarget.size());
                size_t stemLength = std::min(minLen - 2, (size_t)5);  // Use up to 5 chars but leave at least 2 for ending

                if (stemLength > 2 && lastText.substr(0, stemLength) == lastTarget.substr(0, stemLength)) {
                    return 0.85;  // Good match score for different forms of the same word
                }
            }
        }
    }

    // No variant match found
    return 0.0;
}

std::vector<std::string> MLCSAlgorithm::lcs(const std::vector<std::string>& seq1, const std::vector<std::string>& seq2) {
    // Optimization: Empty sequence check
    if (seq1.empty() || seq2.empty()) {
        return {};
    }

    // Optimization: If sequences are very long, use more memory-efficient approach
    if (seq1.size() > 200 || seq2.size() > 200) {
        return lcsEfficient(seq1, seq2);
    }

    // Standard LCS dynamic programming for moderate-sized sequences
    size_t m = seq1.size(), n = seq2.size();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

    // Fill the dp table
    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            if (seq1[i-1] == seq2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = std::max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    // Reconstruct the LCS
    std::vector<std::string> lcs;
    size_t i = m, j = n;

    while (i > 0 && j > 0) {
        if (seq1[i-1] == seq2[j-1]) {
            lcs.push_back(seq1[i-1]);
            --i;
            --j;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            --i;
        } else {
            --j;
        }
    }

    // Reverse the LCS (since we built it backwards)
    std::reverse(lcs.begin(), lcs.end());

    return lcs;
}

std::vector<std::string> MLCSAlgorithm::lcsEfficient(const std::vector<std::string>& seq1, const std::vector<std::string>& seq2) {
    // Ensure seq1 is the shorter sequence for efficiency
    std::vector<std::string> s1 = seq1;
    std::vector<std::string> s2 = seq2;

    if (s1.size() > s2.size()) {
        std::swap(s1, s2);
    }

    size_t m = s1.size(), n = s2.size();

    // Use two rows instead of full matrix
    std::vector<int> current(n + 1, 0);
    std::vector<int> previous(n + 1, 0);

    // Track the choices made for reconstruction
    std::map<std::pair<size_t, size_t>, char> choices;  // (i, j) -> direction

    // Fill the dp table with just two rows
    for (size_t i = 1; i <= m; ++i) {
        previous.swap(current);
        std::fill(current.begin(), current.end(), 0);

        for (size_t j = 1; j <= n; ++j) {
            if (s1[i-1] == s2[j-1]) {
                current[j] = previous[j-1] + 1;
                choices[{i, j}] = 'd';  // Diagonal
            } else if (previous[j] >= current[j-1]) {
                current[j] = previous[j];
                choices[{i, j}] = 'u';  // Up
            } else {
                current[j] = current[j-1];
                choices[{i, j}] = 'l';  // Left
            }
        }
    }

    // Reconstruct the LCS
    std::vector<std::string> lcs;
    size_t i = m, j = n;

    while (i > 0 && j > 0) {
        char direction = choices[{i, j}];

        if (direction == 'd') {
            lcs.push_back(s1[i-1]);
            --i;
            --j;
        } else if (direction == 'u') {
            --i;
        } else {
            --j;
        }
    }

    // Reverse the LCS
    std::reverse(lcs.begin(), lcs.end());

    return lcs;
}

std::vector<std::string> MLCSAlgorithm::findMlcs(const std::vector<std::vector<std::string>>& sequences, int minLength) {
    if (sequences.empty()) {
        return {};
    }

    if (sequences.size() == 1) {
        return sequences[0];
    }

    // For two sequences, use the efficient LCS algorithm
    if (sequences.size() == 2) {
        auto lcsResult = lcs(sequences[0], sequences[1]);
        return lcsResult.size() >= static_cast<size_t>(minLength) ? lcsResult : std::vector<std::string>();
    }

    // For more than two sequences, use optimized approach
    return findMlcsLinear(sequences, minLength);
}

std::vector<std::string> MLCSAlgorithm::findMlcsLinear(const std::vector<std::vector<std::string>>& sequences, int minLength) {
    // Check if we're working with character-level sequences
    bool isCharLevel = true;
    for (const auto& seq : sequences) {
        if (!seq.empty() && seq[0].size() > 1) {
            isCharLevel = false;
            break;
        }
    }

    // For character-level comparison (typical in concept deduplication)
    if (isCharLevel) {
        return findMlcsCharacter(sequences, minLength);
    }

    // For token-level comparison (typical in concept signature extraction)
    return findMlcsToken(sequences, minLength);
}

std::vector<std::string> MLCSAlgorithm::findMlcsCharacter(const std::vector<std::vector<std::string>>& sequences, int minLength) {
    if (sequences.empty()) {
        return {};
    }

    // Progressively find common subsequence
    std::vector<std::string> currentLcs = sequences[0];

    for (size_t i = 1; i < sequences.size(); ++i) {
        currentLcs = lcs(currentLcs, sequences[i]);

        // Early termination if LCS becomes too short
        if (currentLcs.size() < static_cast<size_t>(minLength)) {
            return {};
        }
    }

    return currentLcs.size() >= static_cast<size_t>(minLength) ? currentLcs : std::vector<std::string>();
}

std::vector<std::string> MLCSAlgorithm::findMlcsToken(const std::vector<std::vector<std::string>>& sequences, int minLength) {
    // Calculate the frequency of each token in all sequences
    std::set<std::string> allTokens;
    for (const auto& seq : sequences) {
        allTokens.insert(seq.begin(), seq.end());
    }

    // Track token positions in each sequence
    std::map<std::string, std::vector<std::pair<size_t, size_t>>> tokenPositions;
    for (const auto& token : allTokens) {
        tokenPositions[token] = {};
    }

    for (size_t seqIdx = 0; seqIdx < sequences.size(); ++seqIdx) {
        for (size_t pos = 0; pos < sequences[seqIdx].size(); ++pos) {
            const auto& token = sequences[seqIdx][pos];
            tokenPositions[token].push_back({seqIdx, pos});
        }
    }

    // Find tokens that appear in all sequences
    std::vector<std::string> commonTokens;
    for (const auto& [token, positions] : tokenPositions) {
        std::set<size_t> seqIndices;
        for (const auto& [seqIdx, pos] : positions) {
            seqIndices.insert(seqIdx);
        }

        if (seqIndices.size() == sequences.size()) {
            commonTokens.push_back(token);
        }
    }

    if (commonTokens.empty()) {
        // If no common tokens across all sequences, try a more relaxed approach
        // Find tokens that appear in at least half of the sequences
        size_t minSeqCount = std::max(size_t(2), sequences.size() / 2);

        for (const auto& [token, positions] : tokenPositions) {
            std::set<size_t> seqIndices;
            for (const auto& [seqIdx, pos] : positions) {
                seqIndices.insert(seqIdx);
            }

            if (seqIndices.size() >= minSeqCount) {
                commonTokens.push_back(token);
            }
        }

        if (commonTokens.empty()) {
            return {};
        }
    }

    // Extract n-grams from each sequence
    std::map<std::vector<std::string>, std::vector<size_t>> ngrams;

    for (size_t n = static_cast<size_t>(minLength); n <= 10; ++n) {
        for (size_t seqIdx = 0; seqIdx < sequences.size(); ++seqIdx) {
            const auto& seq = sequences[seqIdx];
            if (seq.size() < n) continue;

            for (size_t i = 0; i <= seq.size() - n; ++i) {
                std::vector<std::string> ngram(seq.begin() + i, seq.begin() + i + n);
                ngrams[ngram].push_back(seqIdx);
            }
        }
    }

    // Find n-grams that appear in at least half of the sequences
    size_t minSeqCount = std::max(size_t(2), sequences.size() / 2);
    std::vector<std::pair<std::vector<std::string>, double>> commonNgrams;

    for (const auto& [ngram, seqIndices] : ngrams) {
        std::set<size_t> uniqueSeqIndices(seqIndices.begin(), seqIndices.end());

        if (uniqueSeqIndices.size() >= minSeqCount) {
            // Score the n-gram by length and number of sequences it appears in
            double score = static_cast<double>(ngram.size()) * static_cast<double>(uniqueSeqIndices.size()) / sequences.size();

            // Boost score for domain-specific terms in the ngram
            size_t domainTermCount = 0;
            for (const auto& token : ngram) {
                if (isDomainTerm(token)) {
                    domainTermCount++;
                }
            }

            if (domainTermCount > 0) {
                score *= (1.0 + 0.2 * static_cast<double>(domainTermCount));
            }

            commonNgrams.push_back({ngram, score});
        }
    }

    // Sort by score (higher score first)
    std::sort(commonNgrams.begin(), commonNgrams.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // Return the highest scoring n-gram if any
    return commonNgrams.empty() ? std::vector<std::string>() : commonNgrams[0].first;
}

std::pair<std::vector<std::string>, float> MLCSAlgorithm::extractConceptSignature(
    const std::string& conceptText,
    const std::vector<std::string>& contexts,
    const std::string& language) {

    // Use provided language or default
    std::string lang = language.empty() ? this->language : language;

    // If no contexts, use the concept text itself
    if (contexts.empty()) {
        std::vector<std::string> preprocessed = preprocessText(conceptText, lang);
        return {preprocessed, 0.5};
    }

    // Extract significant sequences from contexts
    std::vector<std::pair<std::vector<std::string>, float>> significantSequences = findSignificantPatterns(
        contexts, 2, std::max(2, static_cast<int>(contexts.size() / 3)), lang
    );

    // If we found significant sequences
    if (!significantSequences.empty()) {
        // Use the highest scoring sequence as the signature pattern
        auto [signaturePattern, score] = significantSequences[0];

        // Ensure the extracted signature actually relates to the concept
        std::vector<std::string> conceptTokens = preprocessText(conceptText, lang);

        // Check if there's overlap between signature pattern and concept tokens
        std::set<std::string> conceptTokenSet(conceptTokens.begin(), conceptTokens.end());
        bool hasOverlap = false;

        for (const auto& token : signaturePattern) {
            if (conceptTokenSet.find(token) != conceptTokenSet.end()) {
                hasOverlap = true;
                break;
            }
        }

        if (hasOverlap || signaturePattern.size() <= 2) {
            // Calculate confidence based on score
            float confidence = std::min(static_cast<float>(score / 10.0), 0.95f);  // Normalize confidence
            return {signaturePattern, confidence};
        }

        // If no overlap, try the next highest scoring sequence
        if (significantSequences.size() > 1) {
            auto [nextSignaturePattern, nextScore] = significantSequences[1];
            float confidence = std::min(static_cast<float>(nextScore / 10.0), 0.9f);  // Slightly lower confidence
            return {nextSignaturePattern, confidence};
        }
    }

    // If no significant sequences found or no good match, use the preprocessed concept text
    std::vector<std::string> preprocessed = preprocessText(conceptText, lang);
    return {preprocessed, 0.5};
}

std::vector<std::pair<std::vector<std::string>, float>> MLCSAlgorithm::findSignificantPatterns(
    const std::vector<std::string>& texts,
    int minLength,
    int minFrequency,
    const std::string& language) {

    // Use provided language or default
    std::string lang = language.empty() ? this->language : language;

    // Preprocess texts
    std::vector<std::vector<std::string>> preprocessedTexts;
    for (const auto& text : texts) {
        auto tokens = preprocessText(text, lang);
        if (tokens.size() >= static_cast<size_t>(minLength)) {
            preprocessedTexts.push_back(tokens);
        }
    }

    if (preprocessedTexts.empty()) {
        return {};
    }

    // Find common n-grams across texts
    std::map<std::vector<std::string>, int> ngramCounts;

    for (const auto& tokens : preprocessedTexts) {
        std::set<std::vector<std::string>> textNgrams;  // Use set to avoid counting duplicates within the same text

        for (size_t n = static_cast<size_t>(minLength); n <= std::min(size_t(10), tokens.size()); ++n) {
            for (size_t i = 0; i <= tokens.size() - n; ++i) {
                std::vector<std::string> ngram(tokens.begin() + i, tokens.begin() + i + n);
                textNgrams.insert(ngram);
            }
        }

        // Count each unique n-gram once per text
        for (const auto& ngram : textNgrams) {
            ngramCounts[ngram]++;
        }
    }

    // Filter by frequency and sort by score
    std::vector<std::pair<std::vector<std::string>, float>> significantNgrams;

    for (const auto& [ngram, count] : ngramCounts) {
        if (count >= minFrequency) {
            // Calculate base score based on length and frequency
            float lengthWeight = ngram.size() * 0.3f;
            float frequencyWeight = (static_cast<float>(count) / texts.size()) * 2.0f;

            // Count domain terms in the n-gram
            size_t domainTermCount = 0;
            for (const auto& term : ngram) {
                if (isDomainTerm(term)) {
                    domainTermCount++;
                }
            }
            float domainWeight = domainTermCount * 0.5f;

            // Language-specific scoring adjustments
            if (lang == "ru") {
                // For Russian, give higher weights to multi-word terms
                // that correspond to important physics concepts
                if (ngram.size() >= 2) {
                    // Check for important Russian physics bigrams/trigrams
                    std::string term;
                    for (const auto& token : ngram) {
                        if (!term.empty()) term += " ";
                        term += token;
                    }

                    static const std::vector<std::string> importantTerms = {
                        "волновая функция", "квантовая механика", "собственное значение",
                        "собственное состояние", "гильбертово пространство", "принцип неопределенности",
                        "оператор энергии", "оператор импульса", "оператор координаты",
                        "эрмитов оператор", "унитарное преобразование", "стационарное состояние",
                        "квантовая теория", "вакуумное состояние", "матрица плотности",
                        "квантовый осциллятор", "уравнение шредингера"
                    };

                    for (const auto& important : importantTerms) {
                        if (term.find(important) != std::string::npos) {
                            domainWeight += 1.0f;  // Significant boost
                            break;
                        }
                    }
                }
            }

            // Calculate final score
            float score = lengthWeight + frequencyWeight + domainWeight;

            significantNgrams.push_back({ngram, score});
        }
    }

    // Sort by score
    std::sort(significantNgrams.begin(), significantNgrams.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    return significantNgrams;
}
