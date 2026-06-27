package tensor

import (
	"runtime"
	"sync"
)

type Tensor struct {
	Data  []float64
	Shape []int
}

func NewTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &Tensor{
		Data:  make([]float64, size),
		Shape: shape,
	}
}

func Zero(t *Tensor) {
	for i := range t.Data {
		t.Data[i] = 0
	}
}

func MatMul(a, b, c *Tensor) {
	matMulImpl(a, b, c, false, false)
}

func MatMulTranspose(a, b, c *Tensor, transposeA, transposeB bool) {
	matMulImpl(a, b, c, transposeA, transposeB)
}

func matMulImpl(a, b, c *Tensor, transposeA, transposeB bool) {
	aRows := a.Shape[0]
	aCols := a.Shape[1]
	if transposeA {
		aRows, aCols = aCols, aRows
	}
	bRows := b.Shape[0]
	bCols := b.Shape[1]
	if transposeB {
		bRows, bCols = bCols, bRows
	}

	m := aRows
	k := aCols
	n := bCols

	// Zero output.
	for i := range c.Data {
		c.Data[i] = 0
	}

	numWorkers := runtime.NumCPU()
	if m < 32 {
		numWorkers = 1
	}

	var wg sync.WaitGroup
	rowsPerWorker := (m + numWorkers - 1) / numWorkers

	aStoredCols := a.Shape[1]
	bStoredCols := b.Shape[1]

	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := (w + 1) * rowsPerWorker
		if endRow > m {
			endRow = m
		}
		if startRow >= m {
			break
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			switch {
			case !transposeA && !transposeB:
				for i := start; i < end; i++ {
					cRow := c.Data[i*n : i*n+n]
					for kk := 0; kk < k; kk++ {
						aik := a.Data[i*k+kk]
						bRow := b.Data[kk*n : kk*n+n]
						for j, bv := range bRow {
							cRow[j] += aik * bv
						}
					}
				}

			case transposeA && !transposeB:
				aColBuf := make([]float64, k)
				for i := start; i < end; i++ {
					for kk := 0; kk < k; kk++ {
						aColBuf[kk] = a.Data[kk*aStoredCols+i]
					}
					cRow := c.Data[i*n : i*n+n]
					for kk := 0; kk < k; kk++ {
						aik := aColBuf[kk]
						bRow := b.Data[kk*n : kk*n+n]
						for j, bv := range bRow {
							cRow[j] += aik * bv
						}
					}
				}

			case !transposeA && transposeB:
				for i := start; i < end; i++ {
					aRow := a.Data[i*k : i*k+k]
					cRow := c.Data[i*n : i*n+n]
					for j := 0; j < n; j++ {
						bRow := b.Data[j*bStoredCols : j*bStoredCols+k]
						var sum float64
						for kk, av := range aRow {
							sum += av * bRow[kk]
						}
						cRow[j] = sum
					}
				}

			case transposeA && transposeB:
				aColBuf := make([]float64, k)
				for i := start; i < end; i++ {
					for kk := 0; kk < k; kk++ {
						aColBuf[kk] = a.Data[kk*aStoredCols+i]
					}
					cRow := c.Data[i*n : i*n+n]
					for j := 0; j < n; j++ {
						bRow := b.Data[j*bStoredCols : j*bStoredCols+k]
						var sum float64
						for kk, av := range aColBuf {
							sum += av * bRow[kk]
						}
						cRow[j] = sum
					}
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()
}