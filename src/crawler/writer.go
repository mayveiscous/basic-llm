package crawler

import (
	"os"
)

// write final output
// to the dataset file 
// for the model to read
func WriteSamples(path string, samples []Sample) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, s := range samples {
		formatted := FormatSample(s)
		f.WriteString(formatted + "\n")
	}

	return nil
}