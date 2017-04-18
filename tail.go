package lazyseq

import (
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

type tailRes struct {
	In       Seq
	Out      anyvec.Vector
	SeqLens  []int
	OutSize  int
	NumSteps int
}

// Tail creates a packed vector containing the last
// timestep of each sequence in the batch.
//
// Empty sequences are ignored.
//
// The vector size must be the same at every timestep.
// In other words, there must be some number N such that
// there are numPresent*N components in the packed output
// vectors at every timestep.
func Tail(seq Seq) anydiff.Res {
	res := &tailRes{In: seq}

	var lastBatch *anyseq.Batch
	var outs []anyvec.Vector

	handleBatch := func(batch *anyseq.Batch) {
		if lastBatch == nil {
			res.OutSize = batch.Packed.Len() / batch.NumPresent()
			res.SeqLens = make([]int, len(batch.Present))
			outs = make([]anyvec.Vector, len(batch.Present))
		} else {
			for i, p := range batch.Present {
				if !p && lastBatch.Present[i] {
					res.SeqLens[i] = res.NumSteps
					start, end := seqRangeInBatch(lastBatch, i)
					outs[i] = lastBatch.Packed.Slice(start, end)
				}
			}
		}
		lastBatch = batch
	}

	for batch := range seq.Forward() {
		handleBatch(batch)
		res.NumSteps++
	}

	if res.NumSteps == 0 {
		return anydiff.NewConst(seq.Creator().MakeVector(0))
	}

	// Terminate the last sequence(s).
	handleBatch(&anyseq.Batch{Present: make([]bool, len(outs))})

	res.Out = concatSparse(seq.Creator(), outs)
	return res
}

func (t *tailRes) Output() anyvec.Vector {
	return t.Out
}

func (t *tailRes) Vars() anydiff.VarSet {
	return t.In.Vars()
}

func (t *tailRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	downstream := make(chan *anyseq.Batch, 1)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(downstream)

		uVecs := t.splitUpstreamPerSeq(u)
		for time := t.NumSteps - 1; time >= 0; time-- {
			upBatch := t.zeroBatch(time)
			for seqIdx, length := range t.SeqLens {
				if length == time+1 {
					start, end := seqRangeInBatch(upBatch, seqIdx)
					upBatch.Packed.Slice(start, end).Set(uVecs[seqIdx])
				}
			}
			downstream <- upBatch
		}
	}()

	t.In.Propagate(downstream, NewGrad(g))
	wg.Wait()
}

func (t *tailRes) splitUpstreamPerSeq(u anyvec.Vector) []anyvec.Vector {
	res := make([]anyvec.Vector, len(t.SeqLens))
	start := 0
	for i, length := range t.SeqLens {
		if length == 0 {
			continue
		}
		res[i] = u.Slice(start, start+t.OutSize)
		start += t.OutSize
	}
	return res
}

func (t *tailRes) zeroBatch(time int) *anyseq.Batch {
	present := make([]bool, len(t.SeqLens))
	numPres := 0
	for i, length := range t.SeqLens {
		if length > time {
			present[i] = true
			numPres++
		}
	}
	return &anyseq.Batch{
		Present: present,
		Packed:  t.In.Creator().MakeVector(numPres * t.OutSize),
	}
}

func seqRangeInBatch(batch *anyseq.Batch, seqIdx int) (start, end int) {
	vecSize := batch.Packed.Len() / batch.NumPresent()
	for _, p := range batch.Present[:seqIdx] {
		if p {
			start += vecSize
		}
	}
	return start, start + vecSize
}

func concatSparse(c anyvec.Creator, sparse []anyvec.Vector) anyvec.Vector {
	var nonNil []anyvec.Vector
	for _, x := range sparse {
		if x != nil {
			nonNil = append(nonNil, x)
		}
	}
	return c.Concat(nonNil...)
}
