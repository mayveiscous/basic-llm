package tokenizer

import "strings"

type Tokenizer struct {
	CharToID map[string]int
	IDToChar map[int]string
}

func NewTokenizer(text string) *Tokenizer {
	cToID := make(map[string]int)
	IDToC := make(map[int]string)
	id := 0

	// grab unique chars
	for _, char := range text {
		strChar := string(char)

		// if this char isn't already mapped
		// map it
		if _, exists := cToID[strChar]; !exists {
			cToID[strChar] = id
			IDToC[id] = strChar
			id++
		}
	}

	// build tokenizer object
	return &Tokenizer{
		CharToID: cToID,
		IDToChar: IDToC,
	}
}

// convert string characters
// to ids and insert into tokens
func (t *Tokenizer) Encode(text string) []int {
	tokens := make([]int, len(text))


	for i, char := range text {
		tokens[i] = t.CharToID[string(char)]
	}

	return tokens
}

func (t *Tokenizer) Decode(tokens []int) string {
	var sb strings.Builder

	for _, id := range tokens {
		sb.WriteString(t.IDToChar[id])
	}
	
	return sb.String()
}