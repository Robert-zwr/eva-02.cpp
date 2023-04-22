#include <string>
#include <regex>

std::string whitespace_clean(std::string text) {
    // Replace one or more whitespace characters with a single space character
    std::regex pattern("\\s+");
    std::string replacement(" ");
    std::string cleaned_text = std::regex_replace(text, pattern, replacement);
    
    // Remove leading and trailing whitespace characters
    cleaned_text.erase(0, cleaned_text.find_first_not_of(" "));
    cleaned_text.erase(cleaned_text.find_last_not_of(" ") + 1);
    
    return cleaned_text;
}
