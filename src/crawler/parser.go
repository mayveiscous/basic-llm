package crawler

import (
	"regexp"
	"strings"
)

var (
	tagRe = regexp.MustCompile(`<[^>]*>`)
	spaceRe = regexp.MustCompile(`\s+`)
)

// very simple regex matching
// to remove html and whitespace
func StripHTML(html string) string {
	text := tagRe.ReplaceAllString(html, " ")
	text = spaceRe.ReplaceAllString(text, " ")
	return strings.TrimSpace(text)
}