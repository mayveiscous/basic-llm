package crawler

import "strings"

func CleanText(text string) string {
	text = strings.TrimSpace(text)

	// remove extremely short text
	if len(text) < 50 {
		return ""
	}

	badFragments := []string{
		"cookie", "privacy policy", "subscribe", "advertisement",
	}

	lower := strings.ToLower(text)
	_ = lower
	
	// remove nav-like noise
	for _, b := range badFragments {
		text = strings.ReplaceAll(text, b, "")
		lower = strings.ToLower(text)
	}

	text = strings.TrimSpace(text)

	if len(text) < 50 {
		return ""
	}

	return text
}