package tokenizer

import (
	"strings"
)

type Tokenizer struct {
	Vocab         map[string]int
	InverseVocab  map[int]string
	Merges        map[string]string
	MergeOrder    []string
	SpecialTokens map[string]int
}

const sep = "\x00"

// init vocab from corpus
func NewTokenizer(text string) *Tokenizer {
	vocab := make(map[string]int)
	inv := make(map[int]string)

	id := 0

	// define special tokens FIRST
	special := []string{
		"<PAD>",
		"<NL>",
		"<User>",
		"<Assistant>",
	}

	specialMap := make(map[string]int)

	for _, tok := range special {
		vocab[tok] = id
		inv[id] = tok
		specialMap[tok] = id
		id++
	}

	// normal character vocab
	for _, r := range text {
		s := string(r)
		if _, ok := vocab[s]; !ok {
			vocab[s] = id
			inv[id] = s
			id++
		}
	}

	return &Tokenizer{
		Vocab:         vocab,
		InverseVocab:  inv,
		Merges:        make(map[string]string),
		MergeOrder:    []string{},
		SpecialTokens: specialMap,
	}
}

func toSymbols(text string) []string {
	out := make([]string, 0, len(text))
	for _, r := range text {
		out = append(out, string(r))
	}
	return out
}

// count pair frequencies globally
func getPairCounts(corpus [][]string) map[string]int {
	counts := make(map[string]int)
	for _, seq := range corpus {
		for i := 0; i < len(seq)-1; i++ {
			pair := seq[i] + sep + seq[i+1]
			counts[pair]++
		}
	}
	return counts
}

// apply merge to entire corpus
func applyMerge(corpus [][]string, pair string, merged string) [][]string {
	parts := strings.SplitN(pair, sep, 2)
	out := make([][]string, len(corpus))

	for i, seq := range corpus {
		var newSeq []string
		for j := 0; j < len(seq); j++ {
			if j < len(seq)-1 &&
				seq[j] == parts[0] &&
				seq[j+1] == parts[1] {
				newSeq = append(newSeq, merged)
				j++
			} else {
				newSeq = append(newSeq, seq[j])
			}
		}
		out[i] = newSeq
	}

	return out
}

// train bpe
func (t *Tokenizer) Train(text string, numMerges int) {
	corpus := [][]string{}
	symbols := toSymbols(text)
	corpus = append(corpus, symbols)

	nextID := len(t.Vocab)

	for i := 0; i < numMerges; i++ {
		pairCounts := getPairCounts(corpus)

		bestPair := ""
		bestCount := 0

		for p, c := range pairCounts {
			if c > bestCount {
				bestCount = c
				bestPair = p
			}
		}

		if bestCount == 0 {
			break
		}

		parts := strings.SplitN(bestPair, sep, 2)
		merged := parts[0] + parts[1]

		t.Merges[bestPair] = merged
		t.MergeOrder = append(t.MergeOrder, bestPair)

		t.Vocab[merged] = nextID
		t.InverseVocab[nextID] = merged
		nextID++

		corpus = applyMerge(corpus, bestPair, merged)
	}
}

// encode text into token ids
func (t *Tokenizer) Encode(text string) []int {
	symbols := toSymbols(text)

	for _, pair := range t.MergeOrder {
		parts := strings.SplitN(pair, sep, 2)
		if len(parts) != 2 {
			continue
		}

		merged := t.Merges[pair]
		var newSymbols []string

		for i := 0; i < len(symbols); i++ {
			if i < len(symbols)-1 &&
				symbols[i] == parts[0] &&
				symbols[i+1] == parts[1] {
				newSymbols = append(newSymbols, merged)
				i++
			} else {
				newSymbols = append(newSymbols, symbols[i])
			}
		}

		symbols = newSymbols
	}

	var tokens []int
	for _, s := range symbols {
		if id, ok := t.Vocab[s]; ok {
			tokens = append(tokens, id)
		}
	}

	return tokens
}

// decode tokens back to text
func (t *Tokenizer) Decode(tokens []int) string {
	var sb strings.Builder

	for _, id := range tokens {
		if s, ok := t.InverseVocab[id]; ok {
			if s == "<NL>" {
				sb.WriteString("\n")
			} else {
				sb.WriteString(s)
			}
		}
	}

	return sb.String()
}