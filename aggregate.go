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
		In:  in,
		Out: in.Creator().MakeVector(0),
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

type sumEachRes struct {
	In       Seq
	Out      anyvec.Vector
	SeqLens  []int
	MaxLen   int
	NonEmpty []bool
}

// SumEach sums the outputs for each sequence.
// The result is a packed vector with one sum per
// non-empty sequence.
//
// All timesteps must have the same output size.
func SumEach(in Seq) anydiff.Res {
	res := &sumEachRes{
		In:  in,
		Out: in.Creator().MakeVector(0),
	}
	for batch := range in.Forward() {
		if res.MaxLen == 0 {
			res.SeqLens = make([]int, len(batch.Present))
			res.Out = batch.Packed.Copy()
			res.NonEmpty = batch.Present
		} else {
			res.Out.Add(batch.Expand(res.NonEmpty).Packed)
		}

		res.MaxLen++
		for i, pres := range batch.Present {
			if pres {
				res.SeqLens[i] = res.MaxLen
			}
		}
	}
	return res
}

func (s *sumEachRes) Output() anyvec.Vector {
	return s.Out
}

func (s *sumEachRes) Vars() anydiff.VarSet {
	return s.In.Vars()
}

func (s *sumEachRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	uBatch := &anyseq.Batch{Present: s.NonEmpty, Packed: u}
	grad := NewGrad(g)

	downstream := make(chan *anyseq.Batch, 1)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		s.In.Propagate(downstream, grad)
		wg.Done()
	}()

	for i := s.MaxLen - 1; i >= 0; i-- {
		pres := s.presentAtTime(i)
		downstream <- uBatch.Reduce(pres)
	}

	close(downstream)
	wg.Wait()
}

func (s *sumEachRes) presentAtTime(t int) []bool {
	res := make([]bool, len(s.SeqLens))
	for i, l := range s.SeqLens {
		if l > t {
			res[i] = true
		}
	}
	return res
}
