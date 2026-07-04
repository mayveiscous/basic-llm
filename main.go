package main

import (
	"fmt"
	modelPack "llm/src/model"
	"llm/src/tensor"
	"llm/src/tokenizer"
	"llm/src/crawler"
	"math"
	"math/rand"
	"os"
)

func read() []crawler.Sample {
	urls := []string{
		"https://en.wikipedia.org/wiki/Artificial_intelligence",
		"https://en.wikipedia.org/wiki/Neural_network",
		"https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
	}

	var allSamples []crawler.Sample

	for _, url := range urls {
		html, err := crawler.FetchURL(url)
		if err != nil {
   		 	fmt.Println("fetch failed:", url, err)
    		continue
		}

		text := crawler.StripHTML(html)
		text = crawler.CleanText(text)

		if text == "" {
			continue
		}

		if len(text) > 200 {
			allSamples = append(allSamples, crawler.Sample{
				User: text[:len(text)/2],
				Assistant: text[len(text)/2:],
			})
		}
	}

	err := crawler.WriteSamples("src/data/dataset.txt", allSamples)
	if err != nil {
		fmt.Println(err)
	}

	return allSamples
}

func main() {
	trainingPath := "src/data/dataset.txt"
	seqLen := 64
	batchSize := 4
	embeddingDim := 128

	// training settings
	learningRate := 0.002
	epochs := 1000

	fmt.Println("Crawling web...")
	samples := read()

	// debugging values
	var lossSum float64
	var lossCount int

	// read training data
	bytes, err := os.ReadFile(trainingPath)
	if err != nil {
		panic(err)
	}

	source := string(bytes)

	// merge web + local dataset
	for _, s := range samples {
		source += "\n<User> " + s.User + "\n<Assistant> " + s.Assistant + "\n"
	}

	if len(source) < 1000 {
		panic("dataset too small after merging crawler + file")
	}

	tkzr := tokenizer.NewTokenizer(source)

	// train tokenizer

	// O(merges * corpus)
	// no priority queue
	// production systems should use a heap + incremental count updates.
	numMerges := 1000
	fmt.Println("Training BPE Tokenizer...")
	tkzr.Train(source, numMerges)
	fmt.Printf("BPE Training complete. Final Vocab Size: %d\n", len(tkzr.Vocab))
	
	allTokens := tkzr.Encode(source)

	if len(allTokens) < seqLen+2 {
		panic("token stream too small for training")
	}

	vocabSize := len(tkzr.Vocab)

	// build model
	cfg := modelPack.GPTConfig{
		VocabSize:    vocabSize,
		SeqLen:       seqLen,
		EmbeddingDim: embeddingDim,
	}
	model := modelPack.NewGPTModel(cfg)
	ffnDim := embeddingDim * 4
	numTokens := batchSize * seqLen

	// prebuild tensors & buffers
	dLogits := tensor.NewTensor([]int{numTokens, vocabSize})
	dWlm := tensor.NewTensor([]int{embeddingDim, vocabSize})
	dFfnOut := tensor.NewTensor([]int{numTokens, embeddingDim})
	dW2 := tensor.NewTensor([]int{ffnDim, embeddingDim})
	dW1 := tensor.NewTensor([]int{embeddingDim, ffnDim})
	dWq := tensor.NewTensor([]int{embeddingDim, embeddingDim})
	dWk := tensor.NewTensor([]int{embeddingDim, embeddingDim})
	dWv := tensor.NewTensor([]int{embeddingDim, embeddingDim})
	dGamma1 := tensor.NewTensor([]int{embeddingDim})
	dBeta1 := tensor.NewTensor([]int{embeddingDim})
	dGamma2 := tensor.NewTensor([]int{embeddingDim})
	dBeta2 := tensor.NewTensor([]int{embeddingDim})
	dLN1In := tensor.NewTensor([]int{numTokens, embeddingDim})
	dLN2In := tensor.NewTensor([]int{numTokens, embeddingDim})
	dAttnResidualSkip := tensor.NewTensor([]int{numTokens, embeddingDim})

	batchX := make([]int, numTokens)
	batchY := make([]int, numTokens)
	buf := modelPack.NewBuffers(cfg, batchSize)

	fmt.Println("Starting training loop...")
	for step := range epochs {
		// zero values
		tensor.Zero(dLogits)
		tensor.Zero(dWlm)
		tensor.Zero(dFfnOut)
		tensor.Zero(dW2)
		tensor.Zero(dW1)
		tensor.Zero(dWq)
		tensor.Zero(dWk)
		tensor.Zero(dWv)
		tensor.Zero(dGamma1)
		tensor.Zero(dBeta1)
		tensor.Zero(dGamma2)
		tensor.Zero(dBeta2)

		mask := make([]float32, numTokens)

		// build batches
		for b := range batchSize {
			maxStart := len(allTokens) - seqLen - 1
			if maxStart <= 0 {
				panic("invalid token range")
			}

			seqStart := rand.Intn(maxStart)

			for t := 0; t < seqLen; t++ {
				flatIdx := b*seqLen + t

				batchX[flatIdx] = allTokens[seqStart+t]
				batchY[flatIdx] = allTokens[seqStart+t+1]

				// last token has no target -> mask it out
				if t == seqLen-1 {
					mask[flatIdx] = 0
				} else {
					mask[flatIdx] = 1
				}
			}
		}

		// calculate loss
		logits, cache := model.Forward(batchX, batchSize, buf)
		loss := tensor.CalculateCrossEntropy(logits, batchY, mask)
		lossSum += loss
		lossCount++

		// debug every 100 steps
		if step%100 == 0 || step == epochs-1 {
			avgLoss := lossSum / float64(lossCount)
			fmt.Printf("Step %d | Avg Loss: %.4f\n", step, avgLoss)
			lossSum = 0
			lossCount = 0
		}

		// calculate backward loss
		tensor.BackwardCrossEntropy(logits, batchY, mask, dLogits)

		// decay lr
		decayedLR := learningRate * math.Exp(-0.00005*float64(step))

		// backward LM head
		model.BackwardLMHead(cache.FfnResidual, dLogits, dWlm, dFfnOut)
		copy(dAttnResidualSkip.Data, dFfnOut.Data)

		dLN2Out := model.BackwardFFN(cache, dFfnOut, dW1, dW2, cache.LN2Out, buf)
		tensor.LayerNormBackward(dLN2Out, cache.AttnResidual, model.Gamma2, cache.LN2Mean, cache.LN2Var, dLN2In, dGamma2, dBeta2, numTokens, embeddingDim)

		for i := range dLN2In.Data {
			dLN2In.Data[i] += dAttnResidualSkip.Data[i]
		}

		dLN1Out := model.BackwardAttention(cache, dLN2In, dWq, dWk, dWv, buf)
		X2D := &tensor.Tensor{Data: cache.XEmbedded.Data, Shape: []int{numTokens, embeddingDim}}
		tensor.LayerNormBackward(dLN1Out, X2D, model.Gamma1, cache.LN1Mean, cache.LN1Var, dLN1In, dGamma1, dBeta1, numTokens, embeddingDim)

		for i := range dLN1In.Data {
			dLN1In.Data[i] += dLN2In.Data[i]
		}

		// backward token/position embedding
		for i, tokenID := range batchX {
			embRowOffset := tokenID * embeddingDim
			posRowOffset := (i % seqLen) * embeddingDim
			tokenRowOffset := i * embeddingDim

			for d := 0; d < embeddingDim; d++ {
				g := dLN1In.Data[tokenRowOffset+d]
				model.EmbeddingWeights.Data[embRowOffset+d] -= decayedLR * g
				model.PosWeights.Data[posRowOffset+d] -= decayedLR * g
			}
		}

		// clip gradients
		clipNorm := 1.0
		for _, grad := range []*tensor.Tensor{dWlm, dW2, dW1, dWq, dWk, dWv, dGamma1, dBeta1, dGamma2, dBeta2} {
			var norm float64
			for _, v := range grad.Data {
				norm += v * v
			}
			norm = math.Sqrt(norm)
			if norm > clipNorm {
				scale := clipNorm / norm
				for i := range grad.Data {
					grad.Data[i] *= scale
				}
			}
		}

		// weight update
		model.UpdateWeights(decayedLR, dWlm, dW2, dW1, dWq, dWk, dWv, dGamma1, dBeta1, dGamma2, dBeta2)
	}

	result := model.Generate(tkzr, "User: hello\nAssistant:", 100, 0.8)
	fmt.Println(result)
}