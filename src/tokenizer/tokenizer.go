package tokenizer

import (
	"strings"
)

type Tokenizer struct {
	Vocab        map[string]int
	InverseVocab map[int]string
	Merges       map[string]string
	MergeOrder   []string
}

func NewTokenizer(text string) *Tokenizer {
	vocab := make(map[string]int)
	invVocab := make(map[int]string)
	id := 0

	for _, char := range text {
		strChar := string(char)
		if _, exists := vocab[strChar]; !exists {
			vocab[strChar] = id
			invVocab[id] = strChar
			id++
		}
	}

	return &Tokenizer{
		Vocab:        vocab,
		InverseVocab: invVocab,
		Merges:       make(map[string]string),
		MergeOrder:   []string{},
	}
}

// train the tokenizer
// to recognize keywords
// that appear often
// (ie. "User:" and "Assistant:")
func (t *Tokenizer) Train(text string, numMerges int) {
	// trim whitespace
	// split words to each character
	words := strings.Split(text, " ")
	splits := make([][]string, len(words))

	// for each word
	for i, word := range words {
		runes := []rune(word)
		splits[i] = make([]string, len(runes))

		// compare this char and next char
		for j, char := range runes {
			splits[i][j] = string(char)
		}
	}

	nextID := len(t.Vocab)

	for range numMerges {
		pairCounts := make(map[string]int)

		// compute pairs
		// from each split
		for _, wordSplit := range splits {
			for j := 0; j < len(wordSplit)-1; j++ {
				pair := wordSplit[j] + " " + wordSplit[j+1]
				pairCounts[pair]++
			}
		}

		// determine best pair
		// based on how often it appears
		var bestPair string
		maxCount := 0
		for pair, count := range pairCounts {
			if count > maxCount {
				maxCount = count
				bestPair = pair
			}
		}

		// all done if no best pairs
		if maxCount == 0 {
			break
		}

		// split and merge
		parts := strings.Split(bestPair, " ")
		mergedStr := parts[0] + parts[1]
		t.Merges[bestPair] = mergedStr
		t.MergeOrder = append(t.MergeOrder, bestPair)

		// add to vocab and move to next word
		t.Vocab[mergedStr] = nextID
		t.InverseVocab[nextID] = mergedStr
		nextID++

		// find occurence of best pair
		// and replace in every word
		for sIdx, wordSplit := range splits {
			var newSplit []string
			for j := 0; j < len(wordSplit); j++ {
				if j < len(wordSplit)-1 && wordSplit[j] == parts[0] && wordSplit[j+1] == parts[1] {
					newSplit = append(newSplit, mergedStr)
					j++
				} else {
					newSplit = append(newSplit, wordSplit[j])
				}
			}
			splits[sIdx] = newSplit
		}
	}
}

// bpe encode text into token ids
func (t *Tokenizer) Encode(text string) []int {
	// normalize whitespace
	// and split into words
	text = strings.ReplaceAll(text, "\n", " \n ")
	words := strings.Fields(text)
	var allTokens []int

	for _, word := range words {
		// convert word into initial character symbols
		runes := []rune(word)
		symbols := make([]string, len(runes))
		for i, char := range runes {
			symbols[i] = string(char)
		}

		// apply learned bpe merges (in training order)
		for _, pair := range t.MergeOrder {
			parts := strings.SplitN(pair, " ", 2)
			if len(parts) != 2 {
				continue
			}

			merged := t.Merges[pair]
			var newSymbols []string
			for i := 0; i < len(symbols); i++ {
				if i < len(symbols)-1 && symbols[i] == parts[0] && symbols[i+1] == parts[1] {
					newSymbols = append(newSymbols, merged)
					i++
				} else {
					newSymbols = append(newSymbols, symbols[i])
				}
			}
			symbols = newSymbols
		}

		// map final symbols to vocab ids
		for _, sym := range symbols {
			if id, exists := t.Vocab[sym]; exists {
				allTokens = append(allTokens, id)
			}
		}

		// perserve word boundaries
		if spaceID, exists := t.Vocab[" "]; exists {
			allTokens = append(allTokens, spaceID)
		}
	}

	return allTokens
}

// take tokens and rebuild text
func (t *Tokenizer) Decode(tokens []int) string {
	var sb strings.Builder

	for _, id := range tokens {
		tokenStr, exists := t.InverseVocab[id]
		if !exists {
			continue
		}
		sb.WriteString(tokenStr)
	}

	return sb.String()
}