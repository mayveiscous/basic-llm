package tensor

// mask future tokens to prevent cheating
func ApplyCausalMask(scores *Tensor, T int) {
	for i := 0; i < T; i++ {
		for j := 0; j < T; j++ {
			if j > i {
				// future token here, set to a ridicoulsly low number
				// so the model avoids it
				scores.Data[i*T+j] = -1e9
			}
		}
	}
}