package model

import (
	"llm/src/tensor"
	"math"
)

func (m *GPTModel) Forward(batchX []int, batchSize int, buf *Buffers) (*tensor.Tensor, *ForwardCache) {
	seqLen := m.Config.SeqLen
	embeddingDim := m.Config.EmbeddingDim
	numTokens := batchSize * seqLen
	cache := &ForwardCache{}

	for i, tokenID := range batchX {
		weightRowOffset := tokenID * embeddingDim
		outputRowOffset := i * embeddingDim
		copy(buf.XEmbedded.Data[outputRowOffset:outputRowOffset+embeddingDim],
			m.EmbeddingWeights.Data[weightRowOffset:weightRowOffset+embeddingDim])
	}

	for b := 0; b < batchSize; b++ {
		for t := 0; t < seqLen; t++ {
			for d := 0; d < embeddingDim; d++ {
				xIdx := (b*seqLen+t)*embeddingDim + d
				posIdx := t*embeddingDim + d
				buf.XEmbedded.Data[xIdx] += m.PosWeights.Data[posIdx]
			}
		}
	}
	cache.XEmbedded = buf.XEmbedded

	X2D := &tensor.Tensor{Data: buf.XEmbedded.Data, Shape: []int{numTokens, embeddingDim}}

	tensor.LayerNorm(X2D, m.Gamma1, m.Beta1, buf.LN1Out, buf.LN1Mean, buf.LN1Var, numTokens, embeddingDim)
	cache.LN1Out = buf.LN1Out
	cache.LN1Mean = buf.LN1Mean
	cache.LN1Var = buf.LN1Var

	tensor.MatMul(buf.LN1Out, m.Wq, buf.Q)
	tensor.MatMul(buf.LN1Out, m.Wk, buf.K)
	tensor.MatMul(buf.LN1Out, m.Wv, buf.V)
	cache.Q = buf.Q
	cache.K = buf.K
	cache.V = buf.V

	scaleFactor := 1.0 / math.Sqrt(float64(embeddingDim))

	for b := 0; b < batchSize; b++ {
		tokenOffset := b * seqLen

		Qb := &tensor.Tensor{
			Data:  buf.Q.Data[tokenOffset*embeddingDim : (tokenOffset+seqLen)*embeddingDim],
			Shape: []int{seqLen, embeddingDim},
		}
		Kb := &tensor.Tensor{
			Data:  buf.K.Data[tokenOffset*embeddingDim : (tokenOffset+seqLen)*embeddingDim],
			Shape: []int{seqLen, embeddingDim},
		}
		Vb := &tensor.Tensor{
			Data:  buf.V.Data[tokenOffset*embeddingDim : (tokenOffset+seqLen)*embeddingDim],
			Shape: []int{seqLen, embeddingDim},
		}

		// transpose K for QK^T
		for r := 0; r < seqLen; r++ {
			for c := 0; c < embeddingDim; c++ {
				buf.KbT.Data[c*seqLen+r] = Kb.Data[r*embeddingDim+c]
			}
		}

		tensor.MatMul(Qb, buf.KbT, buf.ScoreB)

		for i := range buf.ScoreB.Data {
			buf.ScoreB.Data[i] *= scaleFactor
		}

		tensor.ApplyCausalMask(buf.ScoreB, seqLen)

		for r := 0; r < seqLen; r++ {
			tensor.Softmax(buf.ScoreB.Data[r*seqLen : (r+1)*seqLen])
		}

		scoreBlockOffset := b * seqLen * seqLen
		copy(buf.Scores.Data[scoreBlockOffset:scoreBlockOffset+seqLen*seqLen], buf.ScoreB.Data)

		outB := &tensor.Tensor{
			Data:  buf.AttentionOut.Data[tokenOffset*embeddingDim : (tokenOffset+seqLen)*embeddingDim],
			Shape: []int{seqLen, embeddingDim},
		}
		tensor.MatMul(buf.ScoreB, Vb, outB)
	}

	cache.Scores = buf.Scores
	cache.AttentionOut = buf.AttentionOut

	// attnResidual = XEmbedded + AttentionOut
	// we write this into buf.AttnResidual so LayerNorm2 reads from it,
	// and we can recover it during the backward pass.
	for i := range buf.AttnResidual.Data {
		buf.AttnResidual.Data[i] = buf.XEmbedded.Data[i] + buf.AttentionOut.Data[i]
	}
	cache.AttnResidual = buf.AttnResidual

	tensor.LayerNorm(buf.AttnResidual, m.Gamma2, m.Beta2, buf.LN2Out, buf.LN2Mean, buf.LN2Var, numTokens, embeddingDim)
	cache.LN2Out = buf.LN2Out
	cache.LN2Mean = buf.LN2Mean
	cache.LN2Var = buf.LN2Var

	tensor.MatMul(buf.LN2Out, m.W1, buf.FfnHidden)

	copy(buf.FfnPreGelu.Data, buf.FfnHidden.Data)
	cache.FfnPreGelu = buf.FfnPreGelu
	cache.FfnHidden = buf.FfnHidden
	tensor.Gelu(buf.FfnHidden)

	tensor.MatMul(buf.FfnHidden, m.W2, buf.FfnOut)
	cache.FfnOut = buf.FfnOut

	// ffnResidual = AttnResidual + FfnOut
	for i := range buf.FfnResidual.Data {
		buf.FfnResidual.Data[i] = buf.AttnResidual.Data[i] + buf.FfnOut.Data[i]
	}
	cache.FfnResidual = buf.FfnResidual

	tensor.MatMul(buf.FfnResidual, m.Wlm, buf.Logits)

	return buf.Logits, cache
}

func (m *GPTModel) BackwardLMHead(ffnResidual, dLogits, dWlm, dFfnResidual *tensor.Tensor) {
	// dWlm = ffnResidual^T · dLogits
	// dFfnResidual = dLogits · Wlm^T
	tensor.MatMulTranspose(ffnResidual, dLogits, dWlm, true, false)
	tensor.MatMulTranspose(dLogits, m.Wlm, dFfnResidual, false, true)
}

// ffnIn is the input to W1 (i.e. LN2Out from the cache).
// returns dLN2Out: gradient w.r.t. the input of LN2 (= AttnResidual).
func (m *GPTModel) BackwardFFN(cache *ForwardCache, dFfnOut, dW1, dW2, ffnIn *tensor.Tensor, buf *Buffers) *tensor.Tensor {

	tensor.MatMulTranspose(cache.FfnHidden, dFfnOut, dW2, true, false)
	tensor.MatMulTranspose(dFfnOut, m.W2, buf.dFfnHidden, false, true)

	// backprop through GELU: dFfnHidden *= gelu'(FfnPreGelu)
	tensor.GeluDerivative(cache.FfnPreGelu, buf.dFfnHidden)

	tensor.MatMulTranspose(ffnIn, buf.dFfnHidden, dW1, true, false)
	tensor.MatMulTranspose(buf.dFfnHidden, m.W1, buf.dLN2Out, false, true)

	return buf.dLN2Out
}

func (m *GPTModel) BackwardAttention(cache *ForwardCache, dAttentionOut, dWq, dWk, dWv *tensor.Tensor, buf *Buffers) *tensor.Tensor {
	embeddingDim := dAttentionOut.Shape[1]
	seqLen := m.Config.SeqLen
	batchSize := dAttentionOut.Shape[0] / seqLen
	scaleFactor := 1.0 / math.Sqrt(float64(embeddingDim))

	for b := 0; b < batchSize; b++ {
		tokenOffset := b * seqLen
		scoreOffset := b * seqLen * seqLen

		scoresB := &tensor.Tensor{
			Data:  cache.Scores.Data[scoreOffset : scoreOffset+seqLen*seqLen],
			Shape: []int{seqLen, seqLen},
		}
		Vb := &tensor.Tensor{
			Data:  cache.V.Data[tokenOffset*embeddingDim : (tokenOffset+seqLen)*embeddingDim],
			Shape: []int{seqLen, embeddingDim},
		}
		dOutB := &tensor.Tensor{
			Data:  dAttentionOut.Data[tokenOffset*embeddingDim : (tokenOffset+seqLen)*embeddingDim],
			Shape: []int{seqLen, embeddingDim},
		}

		tensor.MatMulTranspose(scoresB, dOutB, buf.dVb, true, false)
		copy(buf.dV.Data[tokenOffset*embeddingDim:], buf.dVb.Data)

		tensor.MatMulTranspose(dOutB, Vb, buf.dScoreB, false, true)

		// backward through softmax
		for r := 0; r < seqLen; r++ {
			probRow := scoresB.Data[r*seqLen : (r+1)*seqLen]
			dScoreRow := buf.dScoreB.Data[r*seqLen : (r+1)*seqLen]

			var sumDot float64
			for c := 0; c < seqLen; c++ {
				sumDot += dScoreRow[c] * probRow[c]
			}
			for c := 0; c < seqLen; c++ {
				dScoreRow[c] = probRow[c] * (dScoreRow[c] - sumDot)
				dScoreRow[c] *= scaleFactor
			}
		}

		Kb := &tensor.Tensor{
			Data:  cache.K.Data[tokenOffset*embeddingDim : (tokenOffset+seqLen)*embeddingDim],
			Shape: []int{seqLen, embeddingDim},
		}
		Qb := &tensor.Tensor{
			Data:  cache.Q.Data[tokenOffset*embeddingDim : (tokenOffset+seqLen)*embeddingDim],
			Shape: []int{seqLen, embeddingDim},
		}

		tensor.MatMul(buf.dScoreB, Kb, buf.dQb)
		copy(buf.dQ.Data[tokenOffset*embeddingDim:], buf.dQb.Data)

		tensor.MatMulTranspose(buf.dScoreB, Qb, buf.dKb, true, false)
		copy(buf.dK.Data[tokenOffset*embeddingDim:], buf.dKb.Data)
	}

	// accumulate weight gradients across the full batch
	tensor.MatMulTranspose(cache.LN1Out, buf.dQ, dWq, true, false)
	tensor.MatMulTranspose(cache.LN1Out, buf.dK, dWk, true, false)
	tensor.MatMulTranspose(cache.LN1Out, buf.dV, dWv, true, false)


	tensor.MatMulTranspose(buf.dQ, m.Wq, buf.dLN1Out, false, true)
	dTemp := &tensor.Tensor{
		Data:  buf.dFfnHidden.Data[:dAttentionOut.Shape[0]*embeddingDim],
		Shape: []int{dAttentionOut.Shape[0], embeddingDim},
	}
	tensor.MatMulTranspose(buf.dK, m.Wk, dTemp, false, true)
	for i := range buf.dLN1Out.Data {
		buf.dLN1Out.Data[i] += dTemp.Data[i]
	}
	tensor.MatMulTranspose(buf.dV, m.Wv, dTemp, false, true)
	for i := range buf.dLN1Out.Data {
		buf.dLN1Out.Data[i] += dTemp.Data[i]
	}

	return buf.dLN1Out
}