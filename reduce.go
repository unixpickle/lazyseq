package lazyseq

import (
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

type reducedTape struct {
	In      Tape
	Present []bool
}

// ReduceTape produces a Tape without the sequences at
// indices where present is false.
func ReduceTape(t Tape, present []bool) Tape {
	return &reducedTape{In: t, Present: present}
}

func (r *reducedTape) Creator() anyvec.Creator {
	return r.In.Creator()
}

func (r *reducedTape) ReadTape(start, end int) <-chan *anyseq.Batch {
	res := make(chan *anyseq.Batch, 1)
	go func() {
		defer close(res)
		for in := range r.In.ReadTape(start, end) {
			subset := append([]bool{}, in.Present...)
			changed := false
			numPresent := 0
			for i, mask := range r.Present {
				if subset[i] {
					if !mask {
						changed = true
						subset[i] = false
					} else {
						numPresent++
					}
				}
			}
			if numPresent == 0 {
				break
			} else if changed {
				res <- in.Reduce(subset)
			} else {
				res <- in
			}
		}
	}()
	return res
}
