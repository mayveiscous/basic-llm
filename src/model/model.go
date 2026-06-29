package model

import (
	"llm/src/tensor"
	"llm/src/tokenizer"
	"math"
	"math/rand"
)


type GPTConfig struct {
	VocabSize    int
	SeqLen       int
	EmbeddingDim int
}

// single-block transformer parmater set
type GPTModel struct {
	Config           GPTConfig
	EmbeddingWeights *tensor.Tensor
	PosWeights       *tensor.Tensor
	Wq, Wk, Wv       *tensor.Tensor
	W1, W2           *tensor.Tensor
	Wlm              *tensor.Tensor
	Gamma1, Beta1    *tensor.Tensor
	Gamma2, Beta2    *tensor.Tensor
}

// backprop required activation cache
type ForwardCache struct {
	XEmbedded       *tensor.Tensor
	Q, K, V         *tensor.Tensor
	Scores          *tensor.Tensor
	AttentionOut    *tensor.Tensor
	AttnResidual    *tensor.Tensor
	FfnPreGelu      *tensor.Tensor
	FfnHidden       *tensor.Tensor
	FfnOut          *tensor.Tensor
	FfnResidual     *tensor.Tensor
	LN1Mean, LN1Var *tensor.Tensor
	LN1Out          *tensor.Tensor
	LN2Mean, LN2Var *tensor.Tensor
	LN2Out          *tensor.Tensor
}

// reusable buffers to reduce allocations during forward/backward
type Buffers struct {
	XEmbedded    *tensor.Tensor
	Q, K, V      *tensor.Tensor
	Scores       *tensor.Tensor
	AttentionOut *tensor.Tensor
	AttnResidual *tensor.Tensor
	LN1Out       *tensor.Tensor
	LN1Mean      *tensor.Tensor
	LN1Var       *tensor.Tensor
	LN2Out       *tensor.Tensor
	LN2Mean      *tensor.Tensor
	LN2Var       *tensor.Tensor
	FfnPreGelu   *tensor.Tensor
	FfnHidden    *tensor.Tensor
	FfnOut       *tensor.Tensor
	FfnResidual  *tensor.Tensor
	Logits       *tensor.Tensor

	KbT    *tensor.Tensor
	ScoreB *tensor.Tensor

	dFfnHidden *tensor.Tensor
	dQ, dK, dV *tensor.Tensor
	dLN2Out    *tensor.Tensor
	dLN1Out    *tensor.Tensor

	dScoreB *tensor.Tensor
	dVb     *tensor.Tensor
	dQb     *tensor.Tensor
	dKb     *tensor.Tensor
}

// xavier-style initailization for linear layers
func fillRandom(data []float64, fanIn int) {
	scale := math.Sqrt(1.0 / float64(fanIn))
	for i := range data {
		data[i] = (rand.Float64()*2.0 - 1.0) * scale
	}
}

// pre-allocated tensor buffers for a given batch size
// avoids allocation inside forward/backward loops
func NewBuffers(cfg GPTConfig, batchSize int) *Buffers {
	ffnDim := cfg.EmbeddingDim * 4
	numTokens := batchSize * cfg.SeqLen

	return &Buffers{
		// main activations
		XEmbedded: tensor.NewTensor([]int{batchSize, cfg.SeqLen, cfg.EmbeddingDim}),

		// attention projections
		Q: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),
		K: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),
		V: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),

		// attention matrix per batch
		Scores: tensor.NewTensor([]int{batchSize * cfg.SeqLen * cfg.SeqLen}),

		AttentionOut: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),
		AttnResidual: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),

		// layer norm outputs
		LN1Out: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),
		LN1Mean: tensor.NewTensor([]int{numTokens}),
		LN1Var: tensor.NewTensor([]int{numTokens}),

		LN2Out: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),
		LN2Mean: tensor.NewTensor([]int{numTokens}),
		LN2Var: tensor.NewTensor([]int{numTokens}),

		// feed-forward expansion
		FfnPreGelu: tensor.NewTensor([]int{numTokens, ffnDim}),
		FfnHidden: tensor.NewTensor([]int{numTokens, ffnDim}),
		FfnOut: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),
		FfnResidual: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),

		// output logits
		Logits: tensor.NewTensor([]int{numTokens, cfg.VocabSize}),

		// attention reshape helpers
		KbT: tensor.NewTensor([]int{cfg.EmbeddingDim, cfg.SeqLen}),
		ScoreB: tensor.NewTensor([]int{cfg.SeqLen, cfg.SeqLen}),

		// gradients
		dFfnHidden: tensor.NewTensor([]int{numTokens, ffnDim}),
		dQ: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),
		dK: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),
		dV: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),
		dLN2Out: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),
		dLN1Out: tensor.NewTensor([]int{numTokens, cfg.EmbeddingDim}),

		dScoreB: tensor.NewTensor([]int{cfg.SeqLen, cfg.SeqLen}),
		dVb: tensor.NewTensor([]int{cfg.SeqLen, cfg.EmbeddingDim}),
		dQb: tensor.NewTensor([]int{cfg.SeqLen, cfg.EmbeddingDim}),
		dKb: tensor.NewTensor([]int{cfg.SeqLen, cfg.EmbeddingDim}),
	}
}


// initialize full transformer with random weights
func NewGPTModel(cfg GPTConfig) *GPTModel {
	ffnDim := cfg.EmbeddingDim * 4

	m := &GPTModel{
		Config: cfg,

		EmbeddingWeights: tensor.NewTensor([]int{cfg.VocabSize, cfg.EmbeddingDim}),
		PosWeights: tensor.NewTensor([]int{cfg.SeqLen, cfg.EmbeddingDim}),

		Wq: tensor.NewTensor([]int{cfg.EmbeddingDim, cfg.EmbeddingDim}),
		Wk: tensor.NewTensor([]int{cfg.EmbeddingDim, cfg.EmbeddingDim}),
		Wv: tensor.NewTensor([]int{cfg.EmbeddingDim, cfg.EmbeddingDim}),

		W1: tensor.NewTensor([]int{cfg.EmbeddingDim, ffnDim}),
		W2: tensor.NewTensor([]int{ffnDim, cfg.EmbeddingDim}),

		Wlm: tensor.NewTensor([]int{cfg.EmbeddingDim, cfg.VocabSize}),

		Gamma1: tensor.NewTensor([]int{cfg.EmbeddingDim}),
		Beta1: tensor.NewTensor([]int{cfg.EmbeddingDim}),
		Gamma2: tensor.NewTensor([]int{cfg.EmbeddingDim}),
		Beta2: tensor.NewTensor([]int{cfg.EmbeddingDim}),
	}

	// weight init
	fillRandom(m.EmbeddingWeights.Data, cfg.VocabSize)
	fillRandom(m.PosWeights.Data, cfg.SeqLen)
	fillRandom(m.Wq.Data, cfg.EmbeddingDim)
	fillRandom(m.Wk.Data, cfg.EmbeddingDim)
	fillRandom(m.Wv.Data, cfg.EmbeddingDim)
	fillRandom(m.W1.Data, cfg.EmbeddingDim)
	fillRandom(m.W2.Data, ffnDim)
	fillRandom(m.Wlm.Data, cfg.EmbeddingDim)

	// stable layer norm start
	for i := range m.Gamma1.Data {
		m.Gamma1.Data[i] = 1.0
		m.Gamma2.Data[i] = 1.0
	}

	return m
}

// SGD weight update step
func (m *GPTModel) UpdateWeights(learningRate float64, dWlm, dW2, dW1, dWq, dWk, dWv, dGamma1, dBeta1, dGamma2, dBeta2 *tensor.Tensor) {
	applyGradient := func(weights, gradients []float64) {
		for i := range weights {
			weights[i] -= learningRate * gradients[i]
		}
	}

	applyGradient(m.Wlm.Data, dWlm.Data)
	applyGradient(m.W2.Data, dW2.Data)
	applyGradient(m.W1.Data, dW1.Data)
	applyGradient(m.Wq.Data, dWq.Data)
	applyGradient(m.Wk.Data, dWk.Data)
	applyGradient(m.Wv.Data, dWv.Data)

	applyGradient(m.Gamma1.Data, dGamma1.Data)
	applyGradient(m.Beta1.Data, dBeta1.Data)
	applyGradient(m.Gamma2.Data, dGamma2.Data)
	applyGradient(m.Beta2.Data, dBeta2.Data)
}

// autoregressive generation loop using sliding context window
func (m *GPTModel) Generate(tkzr *tokenizer.Tokenizer, prompt string, maxNewTokens int, temperature float64) string {
	promptTokens := tkzr.Encode(prompt)

	buf := NewBuffers(m.Config, 1)

	// full sequence being extended autoregressively
	fullSeq := make([]int, len(promptTokens), len(promptTokens)+maxNewTokens)
	copy(fullSeq, promptTokens)

	for range maxNewTokens {

		// take last SeqLen tokens as model context
		window := make([]int, m.Config.SeqLen)

		start := len(fullSeq) - m.Config.SeqLen

		if start < 0 {
			// left pad with zeros
			pad := m.Config.SeqLen - len(fullSeq)
			for j := 0; j < pad; j++ {
				window[j] = 0
			}
			copy(window[pad:], fullSeq)
		} else {
			copy(window, fullSeq[start:])
		}

		logits, _ := m.Forward(window, 1, buf)

		vocabSize := m.Config.VocabSize
		last := logits.Data[(m.Config.SeqLen-1)*vocabSize : m.Config.SeqLen*vocabSize]

		nextToken := sample(last, temperature)
		fullSeq = append(fullSeq, nextToken)
	}

	return tkzr.Decode(fullSeq[len(promptTokens):])
}

// sampling from logits distribution 
// (greedy or temperature-scaled)
func sample(logits []float64, temperature float64) int {
	if temperature == 0 {
		bestIdx := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[bestIdx] {
				bestIdx = i
			}
		}
		return bestIdx
	}
	scaled := make([]float64, len(logits))
	for i, v := range logits {
		scaled[i] = math.Exp(v / temperature)
	}
	var sum float64
	for _, v := range scaled {
		sum += v
	}
	r := rand.Float64() * sum
	var cumulative float64
	for i, v := range scaled {
		cumulative += v
		if r < cumulative {
			return i
		}
	}
	return len(logits) - 1
}