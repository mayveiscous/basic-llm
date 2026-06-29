package tensor

import (
	"math"
)

func Gelu(t *Tensor) {
	for i, val := range t.Data {
		const sqrt20overpi = 0.7978845608
		const coeff = 0.044715
		cube := coeff * val * val * val
		t.Data[i] = 0.5 * val * (1.0 + math.Tanh(sqrt20overpi*(val+cube)))
	}
}

func GeluDerivative(t, dT *Tensor) {
	for i, x := range t.Data {
		const sqrt2OverPi = 0.7978845608
		const coeff = 0.044715
		x3 := x * x * x
		tanhArg := sqrt2OverPi * (x + coeff*x3)
		tanhVal := math.Tanh(tanhArg)
		sech2 := 1.0 - tanhVal*tanhVal
		derivative := 0.5*(1.0+tanhVal) + 0.5*x*sech2*sqrt2OverPi*(1.0+3.0*coeff*x*x)
		dT.Data[i] *= derivative
	}
}

func getMax(slice []float64) float64 {
	maxVal := slice[0]
	for _, val := range slice {
		if val > maxVal {
			maxVal = val
		}
	}
	return maxVal
}

func Softmax(slice []float64) {
	maxVal := getMax(slice)

	var sum float64
	for i, val := range slice {
		slice[i] = math.Exp(val - maxVal)
		sum += slice[i]
	}

	for i := range slice {
		slice[i] /= sum
	}
}

func CalculateCrossEntropy(logits *Tensor, targets []int) float64 {
	numTokens := logits.Shape[0]
	vocabSize := logits.Shape[1]
	var totalLoss float64

	for i := 0; i < numTokens; i++ {
		rowOffset := i * vocabSize
		rowSlice := logits.Data[rowOffset : rowOffset+vocabSize]

		rowCopy := make([]float64, vocabSize)
		copy(rowCopy, rowSlice)

		Softmax(rowCopy)

		correctTargetID := targets[i]
		correctProb := rowCopy[correctTargetID]

		if correctProb < 1e-15 {
			correctProb = 1e-15
		}
		totalLoss += -math.Log(correctProb)
	}
	return totalLoss / float64(numTokens)
}

func BackwardCrossEntropy(logits *Tensor, targets []int, dLogits *Tensor) {
	numTokens := logits.Shape[0]
	vocabSize := logits.Shape[1]

	for i := 0; i < numTokens; i++ {
		rowOffset := i * vocabSize

		probs := make([]float64, vocabSize)
		copy(probs, logits.Data[rowOffset:rowOffset+vocabSize])
		Softmax(probs)

		correctTargetID := targets[i]
		for j := 0; j < vocabSize; j++ {
			if j == correctTargetID {
				dLogits.Data[rowOffset+j] = (probs[j] - 1.0) / float64(numTokens)
			} else {
				dLogits.Data[rowOffset+j] = probs[j] / float64(numTokens)
			}
		}
	}
}

func LayerNorm(x, gamma, beta, out, mean, variance *Tensor, numTokens, dim int) {
	eps := 1e-5
	for i := 0; i < numTokens; i++ {
		rowOffset := i * dim

		var m float64
		for d := 0; d < dim; d++ {
			m += x.Data[rowOffset+d]
		}
		m /= float64(dim)
		mean.Data[i] = m

		var v float64
		for d := 0; d < dim; d++ {
			diff := x.Data[rowOffset+d] - m
			v += diff * diff
		}
		v /= float64(dim)
		variance.Data[i] = v

		invStd := 1.0 / math.Sqrt(v+eps)
		for d := 0; d < dim; d++ {
			xHat := (x.Data[rowOffset+d] - m) * invStd
			out.Data[rowOffset+d] = gamma.Data[d]*xHat + beta.Data[d]
		}
	}
}

func LayerNormBackward(dOut, x, gamma, mean, variance, dX, dGamma, dBeta *Tensor, numTokens, dim int) {
	eps := 1e-5
	for i := 0; i < numTokens; i++ {
		rowOffset := i * dim
		m := mean.Data[i]
		v := variance.Data[i]
		invStd := 1.0 / math.Sqrt(v+eps)

		var dxHatSum, dxHatXhatSum float64
		for d := 0; d < dim; d++ {
			xHat := (x.Data[rowOffset+d] - m) * invStd
			dGamma.Data[d] += dOut.Data[rowOffset+d] * xHat
			dBeta.Data[d] += dOut.Data[rowOffset+d]
			dxHatSum += dOut.Data[rowOffset+d] * gamma.Data[d]
			dxHatXhatSum += dOut.Data[rowOffset+d] * gamma.Data[d] * xHat
		}

		for d := 0; d < dim; d++ {
			xHat := (x.Data[rowOffset+d] - m) * invStd
			dxHat := dOut.Data[rowOffset+d] * gamma.Data[d]
			dX.Data[rowOffset+d] = invStd * (dxHat - dxHatSum/float64(dim) - xHat*dxHatXhatSum/float64(dim))
		}
	}
}