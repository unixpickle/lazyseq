package lazyrnn

import (
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
)

type fixedHSMRes struct {
	In       Rereader
	Out      <-chan *anyseq.Batch
	Interval int
	Block    anyrnn.Block

	// Once Done is closed, the fields beneath it are
	// considered immutable.
	Done     <-chan struct{}
	Saved    []anyrnn.State
	V        anydiff.VarSet
	NumSteps int
}

// FixedHSM uses fixed-interval hidden state memorization
// to apply the RNN block to the sequence.
// Every intervalSize timesteps, a hidden state is saved.
//
// If the sequence length T is known beforehand, the best
// intervalSize is sqrt(T).
// In this case, the algorithm is equivalent to Chen's
// sqrt(T) algorithm, and it will use O(sqrt(T)) memory.
// See https://arxiv.org/abs/1604.06174.
func FixedHSM(intervalSize int, in Rereader, b anyrnn.Block) Seq {
	if intervalSize < 1 {
		panic("invalid interval size")
	}
	outChan := make(chan *anyseq.Batch, 1)
	doneChan := make(chan struct{})
	res := &fixedHSMRes{
		In:       in,
		Out:      outChan,
		Interval: intervalSize,
		Block:    b,
		Done:     doneChan,
		V:        anydiff.VarSet{},
	}
	go res.forward(outChan, doneChan)
	return res
}

func (f *fixedHSMRes) Creator() anyvec.Creator {
	return f.In.Creator()
}

func (f *fixedHSMRes) Forward() <-chan *anyseq.Batch {
	return f.Out
}

func (f *fixedHSMRes) Vars() anydiff.VarSet {
	<-f.Done
	return f.V
}

func (f *fixedHSMRes) Propagate(u <-chan *anyseq.Batch, grad *Grad) {
	f.finishForward()

	var downstream chan *anyseq.Batch
	grad.Use(func(g anydiff.Grad) {
		if g.Intersects(f.In.Vars()) {
			downstream = make(chan *anyseq.Batch, 1)
		}
	})

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		if downstream != nil {
			defer close(downstream)
		}

		var nextGrad anyrnn.StateGrad
		nextIdx := f.NumSteps
		for i := len(f.Saved) - 1; i >= 0; i-- {
			frag := bptt(f.In.Reread(i*f.Interval, nextIdx), f.Block, f.Saved[i])
			nextGrad = frag.Propagate(downstream, u, nextGrad, grad)
			nextIdx = i * f.Interval
		}

		if nextGrad != nil {
			f.propagateStart(nextGrad, grad)
		}

		if _, ok := <-u; ok {
			panic("too many upstream batches")
		}
	}()

	if downstream != nil {
		f.In.Propagate(downstream, grad)
	}

	wg.Wait()
}

func (f *fixedHSMRes) forward(outChan chan<- *anyseq.Batch, doneChan chan<- struct{}) {
	var state anyrnn.State
	for input := range f.In.Forward() {
		if f.NumSteps == 0 {
			state = f.Block.Start(len(input.Present))
		}
		if state.Present().NumPresent() != input.NumPresent() {
			state = state.Reduce(input.Present)
		}
		if f.NumSteps%f.Interval == 0 {
			f.Saved = append(f.Saved, state)
		}
		f.NumSteps++

		res := f.Block.Step(state, input.Packed)
		f.V = anydiff.MergeVarSets(f.V, res.Vars())
		state = res.State()
		outChan <- &anyseq.Batch{Present: input.Present, Packed: res.Output()}
	}

	f.V = anydiff.MergeVarSets(f.V, f.In.Vars())
	close(doneChan)
	close(outChan)
}

func (f *fixedHSMRes) propagateStart(nextGrad anyrnn.StateGrad, grad *Grad) {
	numSeqs := len(nextGrad.Present())
	if nextGrad.Present().NumPresent() != numSeqs {
		allTrue := make(anyrnn.PresentMap, numSeqs)
		for i := range allTrue {
			allTrue[i] = true
		}
		nextGrad = nextGrad.Expand(allTrue)
	}
	grad.Use(func(g anydiff.Grad) {
		f.Block.PropagateStart(nextGrad, g)
	})
}

func (f *fixedHSMRes) finishForward() {
	for _ = range f.Forward() {
	}
}
