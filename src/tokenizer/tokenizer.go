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

func (t *Tokenizer) Train(text string, numMerges int) {
	words := strings.Split(text, " ")
	splits := make([][]string, len(words))
	for i, word := range words {
		runes := []rune(word)
		splits[i] = make([]string, len(runes))
		for j, char := range runes {
			splits[i][j] = string(char)
		}
	}

	nextID := len(t.Vocab)

	for i := 0; i < numMerges; i++ {
		pairCounts := make(map[string]int)

		for _, wordSplit := range splits {
			for j := 0; j < len(wordSplit)-1; j++ {
				pair := wordSplit[j] + " " + wordSplit[j+1]
				pairCounts[pair]++
			}
		}

		var bestPair string
		maxCount := 0
		for pair, count := range pairCounts {
			if count > maxCount {
				maxCount = count
				bestPair = pair
			}
		}

		if maxCount == 0 {
			break
		}

		parts := strings.Split(bestPair, " ")
		mergedStr := parts[0] + parts[1]
		t.Merges[bestPair] = mergedStr
		t.MergeOrder = append(t.MergeOrder, bestPair)

		t.Vocab[mergedStr] = nextID
		t.InverseVocab[nextID] = mergedStr
		nextID++

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

func (t *Tokenizer) Encode(text string) []int {
	text = strings.ReplaceAll(text, "\n", " \n ")
	words := strings.Fields(text)
	var allTokens []int

	for _, word := range words {
		runes := []rune(word)
		symbols := make([]string, len(runes))
		for i, char := range runes {
			symbols[i] = string(char)
		}

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

		for _, sym := range symbols {
			if id, exists := t.Vocab[sym]; exists {
				allTokens = append(allTokens, id)
			}
		}

		if spaceID, exists := t.Vocab[" "]; exists {
			allTokens = append(allTokens, spaceID)
		}
	}

	return allTokens
}

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