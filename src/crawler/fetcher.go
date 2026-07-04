package crawler

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// grab raw html/text with headers preserved
func FetchURL(rawURL string) (string, error) {
	client := &http.Client{
		Timeout: 15 * time.Second,
	}

	req, err := http.NewRequest("GET", rawURL, nil)
	if err != nil {
		return "", err
	}

	// set user agent so wikipedia doesn't block
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; LLM-training-bot/1.0)")

	// prefer readable text
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("http error: %d", resp.StatusCode)
	}

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(b), nil
}

// use REST API to get plain text
func WikipediaPlainText(title string) (string, error) {
	client := &http.Client{Timeout: 15 * time.Second}

	escaped := url.PathEscape(strings.ReplaceAll(title, " ", "_"))
	apiURL := "https://en.wikipedia.org/api/rest_v1/page/plain/" + escaped

	req, _ := http.NewRequest("GET", apiURL, nil)
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; LLM-training-bot/1.0)")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("wiki error: %s", string(body))
	}

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(b), nil
}