package lazyrnn

import (
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

type meanRes struct {
	In      Seq
	Out     anyvec.Vector
	Count   int
	SeqLens []int
	MaxLen  int
}

// Mean computes the mean over every timestep and over
// every sequence.
//
// This requires a certain level of uniformity.
// In particular, the outputs at every timestep must be
// the same length (although the batch sizes needn't
// match).
//
// If the input is empty, an empty vector is returned.
func Mean(in Seq) anydiff.Res {
	res := &meanRes{
		In:      in,
		Out:     in.Creator().MakeVector(0),
		SeqLens: nil,
	}
	for batch := range in.Forward() {
		vecSize := batch.Packed.Len() / batch.NumPresent()
		if res.Count == 0 {
			res.SeqLens = make([]int, len(batch.Present))
			res.Out = in.Creator().MakeVector(vecSize)
		} else if res.Out.Len() != vecSize {
			panic("inconsistent output sizes")
		}

		res.MaxLen++
		for i, pres := range batch.Present {
			if pres {
				res.SeqLens[i] = res.MaxLen
			}
		}

		sum := anyvec.SumRows(batch.Packed, vecSize)
		res.Out.Add(sum)
		res.Count += batch.NumPresent()
	}
	res.Out.Scale(res.Out.Creator().MakeNumeric(1 / float64(res.Count)))
	return res
}

func (m *meanRes) Output() anyvec.Vector {
	return m.Out
}

func (m *meanRes) Vars() anydiff.VarSet {
	return m.In.Vars()
}

func (m *meanRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	grad := NewGrad(g)

	u.Scale(u.Creator().MakeNumeric(1 / float64(m.Count)))

	downstream := make(chan *anyseq.Batch, 1)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		m.In.Propagate(downstream, grad)
		wg.Done()
	}()

	for i := m.MaxLen - 1; i >= 0; i-- {
		batch := &anyseq.Batch{Present: make([]bool, len(m.SeqLens))}
		for j, seqLen := range m.SeqLens {
			if seqLen > i {
				batch.Present[j] = true
			}
		}
		batch.Packed = u.Creator().MakeVector(u.Len() * batch.NumPresent())
		anyvec.AddRepeated(batch.Packed, u)
		downstream <- batch
	}

	close(downstream)
	wg.Wait()
}
