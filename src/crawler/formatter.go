package crawler

import "strings"

type Sample struct {
	User      string
	Assistant string
}

// format question and answers
// to teach the model prompt and response patterns
func FormatSample(s Sample) string {
	return "<User> " + s.User + " <NL>\n" +
		"<Assistant> " + s.Assistant + " <NL>\n"
}

func ExtractQA(text string) []Sample {
	lines := strings.Split(text, "\n")

	var samples []Sample

	for i := 0; i < len(lines)-1; i++ {
		q := strings.TrimSpace(lines[i])
		a := strings.TrimSpace(lines[i+1])

		if len(q) < 10 || len(a) < 10 {
			continue
		}

		// simple heuristic: question-like line
		// very crude approach.
		if strings.Contains(q, "?") {
			samples = append(samples, Sample{
				User:      q,
				Assistant: a,
			})
		}
	}

	return samples
}